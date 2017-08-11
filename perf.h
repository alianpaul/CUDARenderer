#ifndef PERF_H
#define PERF_H

#include <Windows.h>

#define QUERY_PERFORMANCE 1


#if QUERY_PERFORMANCE
#define QUERY_PERFORMANCE_ENTER                                                             \
			    {                                                                                       \
        LARGE_INTEGER QP_qwTicksPerSec ;                                                    \
        QueryPerformanceFrequency( & QP_qwTicksPerSec ) ;                                   \
        const float QP_milliSecondsPerTick = 1000.0f / float( QP_qwTicksPerSec.QuadPart ) ; \
        LARGE_INTEGER QP_iTimeEnter  ;                                                      \
        QueryPerformanceCounter( & QP_iTimeEnter ) ;

#define QUERY_PERFORMANCE_EXIT( strName )                                                                                               \
						{                                                                                                                               \
            LARGE_INTEGER iTimeExit ;                                                                                                   \
            QueryPerformanceCounter( & iTimeExit ) ;                                                                                    \
            LARGE_INTEGER iDuration ;                                                                                                   \
            iDuration.QuadPart = iTimeExit.QuadPart - QP_iTimeEnter.QuadPart ;                                                          \
            const float fDuration = QP_milliSecondsPerTick * float( iDuration.QuadPart ) ;                                              \
            static LARGE_INTEGER strName ## _DurationTotal ;                                                                            \
            strName ## _DurationTotal.QuadPart += iDuration.QuadPart ;                                                                  \
            static unsigned strName ## _Count = 0 ;                                                                                     \
            ++ strName ## _Count ;                                                                                                      \
            const float fDurAvg = QP_milliSecondsPerTick * float( strName ## _DurationTotal.QuadPart ) / float( strName ## _Count ) ;   \
            fprintf( stderr , # strName " = %g (%g) ms\n", fDuration , fDurAvg ) ;														\
						}																															\
			    }
#else
#define QUERY_PERFORMANCE_ENTER
#define QUERY_PERFORMANCE_EXIT( strName )
#endif

#endif