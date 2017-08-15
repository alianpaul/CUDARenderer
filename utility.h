#ifndef UTIL_H
#define UTIL_H

#define MAX2(x, y)	(((x) < (y))? (y) : (x))
#define MIN2(x, y)	(((x) < (y))? (x) : (y))
#define POW2(x)		((x) * (x))
#define POW3(x)		((x) * (x) *(x))
#define CLAMP(x, min, max) MIN2( MAX2(x, min), max)

#endif