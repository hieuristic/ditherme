#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <dirent.h>
#include <sys/stat.h>
#include <pthread.h>
#include <vulkan/vulkan.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK_VK(res) if(res != VK_SUCCESS) { fprintf(stderr, "Vulkan Error: %d\n", res); exit(1); }

typedef struct {
    VkInstance instance; VkPhysicalDevice phys; VkDevice dev; VkQueue q; uint32_t qFam;
    VkPipeline pipe; VkPipelineLayout pl; VkDescriptorSetLayout dsl; VkShaderModule sm;
    pthread_mutex_t q_mtx;
} VkCtx;

typedef struct {
    int qzlvl;
    float ratio;
    float offX;
    float offY;
} PushConstants;

typedef struct {
    char in_path[512]; char out_path[512]; char thumb_path[512];
    int qzlvl; VkCtx* ctx;
} Job;

uint32_t findMem(VkCtx* ctx, uint32_t filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties m; vkGetPhysicalDeviceMemoryProperties(ctx->phys, &m);
    for (uint32_t i = 0; i < m.memoryTypeCount; i++) if ((filter & (1 << i)) && (m.memoryTypes[i].propertyFlags & props) == props) return i;
    return 0;
}

void vk_buf(VkCtx* ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer* b, VkDeviceMemory* m) {
    VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
    CHECK_VK(vkCreateBuffer(ctx->dev, &bi, NULL, b));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(ctx->dev, *b, &req);
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = findMem(ctx, req.memoryTypeBits, props) };
    CHECK_VK(vkAllocateMemory(ctx->dev, &ai, NULL, m));
    vkBindBufferMemory(ctx->dev, *b, *m, 0);
}

void vk_img(VkCtx* ctx, uint32_t w, uint32_t h, VkImageUsageFlags usage, VkImage* img, VkDeviceMemory* mem, VkImageView* view) {
    VkImageCreateInfo ii = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .format = VK_FORMAT_R8G8B8A8_UNORM, .tiling = VK_IMAGE_TILING_OPTIMAL, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .samples = VK_SAMPLE_COUNT_1_BIT };
    CHECK_VK(vkCreateImage(ctx->dev, &ii, NULL, img));
    VkMemoryRequirements req; vkGetImageMemoryRequirements(ctx->dev, *img, &req);
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = findMem(ctx, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
    CHECK_VK(vkAllocateMemory(ctx->dev, &ai, NULL, mem));
    vkBindImageMemory(ctx->dev, *img, *mem, 0);
    VkImageViewCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1} };
    CHECK_VK(vkCreateImageView(ctx->dev, &vi, NULL, view));
}

VkShaderModule load_shader(VkDevice dev, const char* path) {
    FILE* f = fopen(path, "rb"); if(!f) return VK_NULL_HANDLE;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t* code = malloc(sz); fread(code, 1, sz, f); fclose(f);
    VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = code };
    VkShaderModule m; VkResult res = vkCreateShaderModule(dev, &ci, NULL, &m); free(code); return (res == VK_SUCCESS) ? m : VK_NULL_HANDLE;
}

void process_image(Job* job) {
    int w, h, c; unsigned char* data = stbi_load(job->in_path, &w, &h, &c, 4);
    if(!data) return;
    float scale = 0.1f;
    if (w * scale < 256.0f) scale = 256.0f / (float)w;
    if (h * scale < 256.0f) scale = (256.0f / (float)h > scale) ? (256.0f / (float)h) : scale;
    int nw = (int)(w * scale), nh = (int)(h * scale);

    VkCtx* ctx = job->ctx;
    VkBuffer sb_in; VkDeviceMemory sm_in;
    vk_buf(ctx, w*h*4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &sb_in, &sm_in);
    void* m; vkMapMemory(ctx->dev, sm_in, 0, w*h*4, 0, &m); memcpy(m, data, w*h*4); vkUnmapMemory(ctx->dev, sm_in);
    stbi_image_free(data);

    VkImage img_in; VkDeviceMemory im_in; VkImageView iv_in;
    vk_img(ctx, w, h, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT, &img_in, &im_in, &iv_in);

    VkCommandPool cp; VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = ctx->qFam};
    vkCreateCommandPool(ctx->dev, &pci, NULL, &cp);
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb; vkAllocateCommandBuffers(ctx->dev, &cbai, &cb);
    
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    VkImageMemoryBarrier imb = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .image = img_in, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &imb);
    VkBufferImageCopy region = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {(uint32_t)w, (uint32_t)h, 1}};
    vkCmdCopyBufferToImage(cb, sb_in, img_in, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageMemoryBarrier imb2 = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT, .image = img_in, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &imb2);
    vkEndCommandBuffer(cb);
    pthread_mutex_lock(&ctx->q_mtx);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    vkQueueSubmit(ctx->q, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(ctx->q);
    pthread_mutex_unlock(&ctx->q_mtx);

    // Main image pass
    VkImage img_out; VkDeviceMemory im_out; VkImageView iv_out;
    vk_img(ctx, nw, nh, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &img_out, &im_out, &iv_out);
    VkDescriptorPool dp; VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    vkCreateDescriptorPool(ctx->dev, &dpci, NULL, &dp);
    VkDescriptorSet ds; VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &ctx->dsl};
    vkAllocateDescriptorSets(ctx->dev, &dsai, &ds);
    VkDescriptorImageInfo dii[2] = {{0, iv_in, VK_IMAGE_LAYOUT_GENERAL}, {0, iv_out, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wd[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &dii[0]}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &dii[1]}};
    vkUpdateDescriptorSets(ctx->dev, 2, wd, 0, NULL);
    vkBeginCommandBuffer(cb, &bi);
    VkImageMemoryBarrier imb3 = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .image = img_out, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &imb3);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pl, 0, 1, &ds, 0, NULL);
    PushConstants pc_main = { job->qzlvl, (float)w / (float)nw, 0, 0 };
    vkCmdPushConstants(cb, ctx->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_main);
    vkCmdDispatch(cb, (nw+15)/16, (nh+15)/16, 1);
    VkImageMemoryBarrier imb4 = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT, .image = img_out, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &imb4);
    VkBuffer sb_out; VkDeviceMemory sm_out;
    vk_buf(ctx, nw*nh*4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &sb_out, &sm_out);
    VkBufferImageCopy r2 = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {(uint32_t)nw, (uint32_t)nh, 1}};
    vkCmdCopyImageToBuffer(cb, img_out, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sb_out, 1, &r2);
    vkEndCommandBuffer(cb);
    pthread_mutex_lock(&ctx->q_mtx);
    vkQueueSubmit(ctx->q, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(ctx->q);
    pthread_mutex_unlock(&ctx->q_mtx);
    vkMapMemory(ctx->dev, sm_out, 0, nw*nh*4, 0, &m); stbi_write_png(job->out_path, nw, nh, 4, m, nw*4); vkUnmapMemory(ctx->dev, sm_out);
    vkDestroyDescriptorPool(ctx->dev, dp, NULL);
    vkDestroyImageView(ctx->dev, iv_out, NULL); vkDestroyImage(ctx->dev, img_out, NULL); vkFreeMemory(ctx->dev, im_out, NULL);
    vkDestroyBuffer(ctx->dev, sb_out, NULL); vkFreeMemory(ctx->dev, sm_out, NULL);

    // Thumbnail pass
    if (job->thumb_path[0]) {
        VkImage img_t; VkDeviceMemory im_t; VkImageView iv_t;
        vk_img(ctx, 32, 32, VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &img_t, &im_t, &iv_t);
        vkCreateDescriptorPool(ctx->dev, &dpci, NULL, &dp);
        vkAllocateDescriptorSets(ctx->dev, &dsai, &ds);
        VkDescriptorImageInfo dii_t[2] = {{0, iv_in, VK_IMAGE_LAYOUT_GENERAL}, {0, iv_t, VK_IMAGE_LAYOUT_GENERAL}};
        VkWriteDescriptorSet wd_t[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &dii_t[0]}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &dii_t[1]}};
        vkUpdateDescriptorSets(ctx->dev, 2, wd_t, 0, NULL);
        vkBeginCommandBuffer(cb, &bi);
        VkImageMemoryBarrier imb5 = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .image = img_t, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &imb5);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pl, 0, 1, &ds, 0, NULL);
        float r = (w < h) ? (float)w / 32.0f : (float)h / 32.0f;
        PushConstants pc_t = { job->qzlvl, r, (w - r * 32.0f) / 2.0f, (h - r * 32.0f) / 2.0f };
        vkCmdPushConstants(cb, ctx->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc_t);
        vkCmdDispatch(cb, 2, 2, 1);
        VkImageMemoryBarrier imb6 = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT, .image = img_t, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &imb6);
        VkBuffer sb_t; VkDeviceMemory sm_t;
        vk_buf(ctx, 32*32*4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &sb_t, &sm_t);
        VkBufferImageCopy r3 = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {32, 32, 1}};
        vkCmdCopyImageToBuffer(cb, img_t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sb_t, 1, &r3);
        vkEndCommandBuffer(cb);
        pthread_mutex_lock(&ctx->q_mtx);
        vkQueueSubmit(ctx->q, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(ctx->q);
        pthread_mutex_unlock(&ctx->q_mtx);
        vkMapMemory(ctx->dev, sm_t, 0, 32*32*4, 0, &m); stbi_write_png(job->thumb_path, 32, 32, 4, m, 32*4); vkUnmapMemory(ctx->dev, sm_t);
        vkDestroyDescriptorPool(ctx->dev, dp, NULL);
        vkDestroyImageView(ctx->dev, iv_t, NULL); vkDestroyImage(ctx->dev, img_t, NULL); vkFreeMemory(ctx->dev, im_t, NULL);
        vkDestroyBuffer(ctx->dev, sb_t, NULL); vkFreeMemory(ctx->dev, sm_t, NULL);
    }
    vkDestroyCommandPool(ctx->dev, cp, NULL);
    vkDestroyImageView(ctx->dev, iv_in, NULL); vkDestroyImage(ctx->dev, img_in, NULL); vkFreeMemory(ctx->dev, im_in, NULL);
    vkDestroyBuffer(ctx->dev, sb_in, NULL); vkFreeMemory(ctx->dev, sm_in, NULL);
}

typedef struct { Job* jobs; int count; int current; pthread_mutex_t mtx; } Pool;
void* worker(void* arg) {
    Pool* p = (Pool*)arg;
    while(1) {
        pthread_mutex_lock(&p->mtx);
        if(p->current >= p->count) { pthread_mutex_unlock(&p->mtx); break; }
        int idx = p->current++; int curr = p->current;
        pthread_mutex_unlock(&p->mtx);
        
        printf("[%d/%d] Processing: %s\n", curr, p->count, p->jobs[idx].in_path);
        process_image(&p->jobs[idx]);
    }
    return NULL;
}

int main(int argc, char** argv) {
    char *in = NULL, *out = NULL, *thumb = NULL; int qzlvl = 8; int opt;
    static struct option long_options[] = { {"input", 1, 0, 'i'}, {"output", 1, 0, 'o'}, {"thumbnail", 1, 0, 't'}, {"qzlvl", 1, 0, 'q'}, {0, 0, 0, 0} };
    while ((opt = getopt_long(argc, argv, "i:o:t:q:", long_options, NULL)) != -1) {
        if(opt=='i') in=optarg; else if(opt=='o') out=optarg; else if(opt=='t') thumb=optarg; else if(opt=='q') qzlvl=atoi(optarg);
    }
    if (!in || !out) { printf("Usage: %s -i <in_dir> -o <out_dir> [-t <thumb_dir>]\n", argv[0]); return 1; }
    mkdir(out, 0777); if(thumb) mkdir(thumb, 0777);
    VkCtx ctx = {0}; pthread_mutex_init(&ctx.q_mtx, NULL);
    const char* ie[] = {"VK_KHR_portability_enumeration"};
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .enabledExtensionCount = 1, .ppEnabledExtensionNames = ie, .flags = 0x00000001 };
    if (vkCreateInstance(&ici, NULL, &ctx.instance) != VK_SUCCESS) { ici.enabledExtensionCount = 0; ici.flags = 0; CHECK_VK(vkCreateInstance(&ici, NULL, &ctx.instance)); }
    uint32_t n; vkEnumeratePhysicalDevices(ctx.instance, &n, NULL); VkPhysicalDevice* ds = malloc(sizeof(VkPhysicalDevice)*n); vkEnumeratePhysicalDevices(ctx.instance, &n, ds); ctx.phys = ds[0]; free(ds);
    float qp = 1.0f; VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &qp };
    const char* de[] = {"VK_KHR_portability_subset"}; VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    uint32_t en; vkEnumerateDeviceExtensionProperties(ctx.phys, NULL, &en, NULL); VkExtensionProperties* ep = malloc(sizeof(VkExtensionProperties)*en); vkEnumerateDeviceExtensionProperties(ctx.phys, NULL, &en, ep);
    for(uint32_t i=0; i<en; i++) if(!strcmp(ep[i].extensionName, de[0])) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = de; break; }
    vkCreateDevice(ctx.phys, &dci, NULL, &ctx.dev); vkGetDeviceQueue(ctx.dev, 0, 0, &ctx.q); ctx.qFam = 0;
    VkDescriptorSetLayoutBinding lb[2] = { {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL} };
    VkDescriptorSetLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = lb };
    vkCreateDescriptorSetLayout(ctx.dev, &lci, NULL, &ctx.dsl);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &ctx.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    vkCreatePipelineLayout(ctx.dev, &plci, NULL, &ctx.pl);
    ctx.sm = load_shader(ctx.dev, DITHER_SHADER_PATH);
    VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = ctx.pl, .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = ctx.sm, .pName = "main"} };
    vkCreateComputePipelines(ctx.dev, NULL, 1, &cpci, NULL, &ctx.pipe);
    DIR* d = opendir(in); Job* jobs = malloc(sizeof(Job)*1024); int count = 0; struct dirent* dir;
    while((dir = readdir(d))) {
        const char* ext = strrchr(dir->d_name, '.');
        if(!ext || (strcasecmp(ext,".png") && strcasecmp(ext,".jpg") && strcasecmp(ext,".jpeg"))) continue;
        snprintf(jobs[count].in_path, 512, "%s/%s", in, dir->d_name);
        snprintf(jobs[count].out_path, 512, "%s/%s", out, dir->d_name);
        if(thumb) snprintf(jobs[count].thumb_path, 512, "%s/%s", thumb, dir->d_name); else jobs[count].thumb_path[0] = 0;
        jobs[count].qzlvl = qzlvl; jobs[count].ctx = &ctx; count++;
    }
    closedir(d);
    Pool pool = {jobs, count, 0}; pthread_mutex_init(&pool.mtx, NULL);
    pthread_t threads[1]; for(int j=0; j<1; j++) pthread_create(&threads[j], NULL, worker, &pool);
    for(int j=0; j<1; j++) pthread_join(threads[j], NULL);
    printf("\nDone.\n");
    vkDestroyPipeline(ctx.dev, ctx.pipe, NULL); vkDestroyPipelineLayout(ctx.dev, ctx.pl, NULL);
    vkDestroyDescriptorSetLayout(ctx.dev, ctx.dsl, NULL); vkDestroyShaderModule(ctx.dev, ctx.sm, NULL);
    vkDestroyDevice(ctx.dev, NULL); vkDestroyInstance(ctx.instance, NULL);
    return 0;
}
