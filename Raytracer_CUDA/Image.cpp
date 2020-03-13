#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "Image.h"

#define NUM_CHANNELS 3

Image::Image(int width, int height)
{
    _width = width;
    _height = height;
    _imageBuffer.resize(width * height * 3, 0.0f);
}

// set pixel at coordinate (x,y) with specified color
void Image::setPixel(int x, int y, const glm::vec3& color)
{
    int index = y * _width * NUM_CHANNELS + x * NUM_CHANNELS;
    _imageBuffer[index] = color.r;
    _imageBuffer[++index] = color.g;
    _imageBuffer[++index] = color.b;
}

// save image as png and jpg.
void Image::save()
{
    stbi_flip_vertically_on_write(true);
    const uint8_t* imgArray = &(_imageBuffer[0]);
    stbi_write_png("image.png", _width, _height, NUM_CHANNELS, imgArray, _width * NUM_CHANNELS);
    stbi_write_jpg("image.jpg", _width, _height, NUM_CHANNELS, imgArray, 100);
}
