// Antialiased Downsampling + Bayer Dither in Slang

struct PushConstants
{
    int qzlvl;
    float ratio;
    float offX;
    float offY;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0)]]
Texture2D<float4> inputImage;

[[vk::binding(1)]]
RWTexture2D<float4> outputImage;

static const float Bayer8[8][8] = {
    { 0, 48, 12, 60,  3, 51, 15, 63 },
    { 32, 16, 44, 28, 35, 19, 47, 31 },
    { 8, 56,  4, 52, 11, 59,  7, 55 },
    { 40, 24, 36, 20, 43, 27, 39, 23 },
    { 2, 50, 14, 62,  1, 49, 13, 61 },
    { 34, 18, 46, 30, 33, 17, 45, 29 },
    { 10, 58,  6, 54,  9, 57,  5, 53 },
    { 42, 26, 38, 22, 41, 25, 37, 21 }
};

[numthreads(16, 16, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint2 outCoord = threadId.xy;
    uint2 outSize;
    outputImage.GetDimensions(outSize.x, outSize.y);
    uint2 inSize;
    inputImage.GetDimensions(inSize.x, inSize.y);

    if (outCoord.x < outSize.x && outCoord.y < outSize.y)
    {
        // Antialiased Downsampling with Offset & Ratio
        float ratio = pc.ratio;
        float2 start = float2(outCoord) * ratio + float2(pc.offX, pc.offY);
        float2 end = float2(outCoord + 1) * ratio + float2(pc.offX, pc.offY);
        
        float4 sum = 0.0;
        float count = 0.0;
        
        // Simple box filter for antialiasing
        for (int y = int(start.y); y < int(end.y); y++)
        {
            for (int x = int(start.x); x < int(end.x); x++)
            {
                if (x >= 0 && x < int(inSize.x) && y >= 0 && y < int(inSize.y)) {
                    sum += inputImage[uint2(x, y)];
                    count += 1.0;
                }
            }
        }
        
        float4 color = (count > 0.0) ? (sum / count) : inputImage[uint2(clamp(start, float2(0,0), float2(inSize-1)))];
        
        // Bayer Dither
        float bayerValue = Bayer8[outCoord.y % 8][outCoord.x % 8] / 64.0;
        float3 c = color.rgb;
        float spread = 1.0 / float(pc.qzlvl - 1);
        c += (bayerValue - 0.5) * spread;
        
        // Quantize
        c = floor(c * float(pc.qzlvl - 1) + 0.5) / float(pc.qzlvl - 1);
        
        outputImage[outCoord] = float4(c, color.a);
    }
}
