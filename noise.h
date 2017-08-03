#ifndef NOISE_H
#define NOISE_H


void vec2CellNoise(float location[3], float result[2], int index);

void getNoiseTables(int** permX, int** permY, float** value1D);

#endif