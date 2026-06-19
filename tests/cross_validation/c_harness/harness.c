/* Standalone harness: call the compiled XLALSimulateExactPulsarSignal and
 * XLALGeneratePulsarSignal and dump the REAL4TimeSeries for comparison against
 * the ripple JAX implementation.
 *
 * Structs are declared here (no LAL headers available) matching the ABI of
 * lal/lalpulsar; layout is self-verified at runtime (detector location, output
 * epoch/deltaT) before the data are trusted.
 *
 * Usage: ./harness <earth> <sun> <out_exact> <out_gen0> <out_gen_het>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef double REAL8;
typedef float REAL4;
typedef int INT4;
typedef unsigned int UINT4;
typedef short INT2;
typedef unsigned short UINT2;
typedef char CHAR;
enum { LALNameLength = 64 };
#define LALNumUnits 7

typedef struct { INT4 gpsSeconds; INT4 gpsNanoSeconds; } LIGOTimeGPS;
typedef struct {
  INT2 powerOfTen;
  INT2 unitNumerator[LALNumUnits];
  UINT2 unitDenominatorMinusOne[LALNumUnits];
} LALUnit;
typedef struct { UINT4 length; REAL8 *data; } REAL8Vector;
typedef struct { UINT4 length; REAL4 *data; } REAL4Sequence;
typedef struct {
  CHAR name[LALNameLength];
  LIGOTimeGPS epoch;
  REAL8 deltaT;
  REAL8 f0;
  LALUnit sampleUnits;
  REAL4Sequence *data;
} REAL4TimeSeries;

typedef enum {
  COORDINATESYSTEM_HORIZON, COORDINATESYSTEM_GEOGRAPHIC,
  COORDINATESYSTEM_EQUATORIAL, COORDINATESYSTEM_ECLIPTIC,
  COORDINATESYSTEM_GALACTIC
} CoordinateSystem;
typedef struct { REAL8 longitude; REAL8 latitude; CoordinateSystem system; } SkyPosition;

typedef enum {
  LALDETECTORTYPE_ABSENT, LALDETECTORTYPE_IFODIFF, LALDETECTORTYPE_IFOXARM,
  LALDETECTORTYPE_IFOYARM, LALDETECTORTYPE_IFOCOMM, LALDETECTORTYPE_CYLBAR
} LALDetectorType;
typedef struct {
  CHAR name[LALNameLength];
  CHAR prefix[3];
  REAL8 vertexLongitudeRadians;
  REAL8 vertexLatitudeRadians;
  REAL4 vertexElevation;
  REAL4 xArmAltitudeRadians;
  REAL4 xArmAzimuthRadians;
  REAL4 yArmAltitudeRadians;
  REAL4 yArmAzimuthRadians;
  REAL4 xArmMidpoint;
  REAL4 yArmMidpoint;
} LALFrDetector;
typedef struct {
  REAL8 location[3];
  REAL4 response[3][3];
  LALDetectorType type;
  LALFrDetector frDetector;
} LALDetector;

typedef struct tagEphemerisData EphemerisData;            /* opaque */
typedef struct tagCOMPLEX8FrequencySeries COMPLEX8FrequencySeries; /* opaque */

typedef struct {
  struct {
    LIGOTimeGPS refTime;
    SkyPosition position;
    REAL4 psi, aPlus, aCross;
    REAL8 phi0, f0;
    REAL8Vector *spindown;
  } pulsar;
  struct { LIGOTimeGPS tp; REAL8 argp, asini, ecc, period; } orbit;
  REAL8 sourceDeltaT;
  const COMPLEX8FrequencySeries *transfer;
  const LALDetector *site;
  const EphemerisData *ephemerides;
  LIGOTimeGPS startTimeGPS;
  UINT4 duration;
  REAL8 samplingRate;
  REAL8 fHeterodyne;
  UINT4 dtDelayBy2;
  UINT4 dtPolBy2;
} PulsarSignalParams;

extern const LALDetector lalCachedDetectors[];
extern EphemerisData *XLALInitBarycenter(const char *, const char *);
extern REAL8Vector *XLALCreateREAL8Vector(UINT4);
extern REAL4TimeSeries *XLALSimulateExactPulsarSignal(const PulsarSignalParams *);
extern REAL4TimeSeries *XLALGeneratePulsarSignal(const PulsarSignalParams *);

#define LHODIFF 5

/* shared signal parameters (must match the Python comparison) */
static const int    START_GPS = 1000000000;
static const double ALPHA = 1.3, DELTA = -0.5;
static const double F0 = 12.3, F1 = -1.1e-9, F2 = 2.0e-18;
static const double PHI0 = 1.1, PSI = 0.37;
static const double APLUS = 1.0, ACROSS = 0.64;

static void fill_common(PulsarSignalParams *p, const EphemerisData *edat) {
  memset(p, 0, sizeof(*p));
  p->pulsar.refTime.gpsSeconds = 0;     /* => use startTime -> SSB */
  p->pulsar.position.longitude = ALPHA;
  p->pulsar.position.latitude  = DELTA;
  p->pulsar.position.system    = COORDINATESYSTEM_EQUATORIAL;
  p->pulsar.psi = PSI; p->pulsar.aPlus = APLUS; p->pulsar.aCross = ACROSS;
  p->pulsar.phi0 = PHI0; p->pulsar.f0 = F0;
  REAL8Vector *sd = XLALCreateREAL8Vector(2);
  sd->data[0] = F1; sd->data[1] = F2;
  p->pulsar.spindown = sd;
  p->site = &lalCachedDetectors[LHODIFF];
  p->ephemerides = edat;
  p->startTimeGPS.gpsSeconds = START_GPS;
  p->startTimeGPS.gpsNanoSeconds = 0;
}

static void dump(const char *path, const REAL4TimeSeries *ts) {
  FILE *f = fopen(path, "wb");
  UINT4 n = ts->data->length;
  fwrite(&n, sizeof(UINT4), 1, f);
  fwrite(&ts->epoch.gpsSeconds, sizeof(INT4), 1, f);
  fwrite(&ts->epoch.gpsNanoSeconds, sizeof(INT4), 1, f);
  fwrite(&ts->deltaT, sizeof(REAL8), 1, f);
  fwrite(&ts->f0, sizeof(REAL8), 1, f);
  fwrite(ts->data->data, sizeof(REAL4), n, f);
  fclose(f);
  fprintf(stderr, "  wrote %s: n=%u epoch=%d.%09d deltaT=%g f0=%g\n",
          path, n, ts->epoch.gpsSeconds, ts->epoch.gpsNanoSeconds, ts->deltaT, ts->f0);
}

int main(int argc, char **argv) {
  if (argc != 6) { fprintf(stderr, "usage: %s earth sun out_exact out_gen0 out_genhet\n", argv[0]); return 2; }

  /* ----- self-verify struct layout via the detector location ----- */
  const LALDetector *h1 = &lalCachedDetectors[LHODIFF];
  fprintf(stderr, "H1 location read: %.5f %.5f %.5f (expect -2161414.92636 -3834695.17889 4600350.22664)\n",
          h1->location[0], h1->location[1], h1->location[2]);
  if (fabs(h1->location[0] - (-2161414.92636)) > 1e-3) {
    fprintf(stderr, "STRUCT LAYOUT MISMATCH for LALDetector — aborting\n"); return 3;
  }

  EphemerisData *edat = XLALInitBarycenter(argv[1], argv[2]);
  if (!edat) { fprintf(stderr, "XLALInitBarycenter failed\n"); return 4; }

  PulsarSignalParams p;

  /* ----- exact ----- */
  fill_common(&p, edat);
  p.duration = 3600; p.samplingRate = 64.0; p.fHeterodyne = 0.0;
  REAL4TimeSeries *ex = XLALSimulateExactPulsarSignal(&p);
  if (!ex) { fprintf(stderr, "SimulateExact failed\n"); return 5; }
  dump(argv[3], ex);

  /* ----- generate (no heterodyne), fine interpolation tables ----- */
  fill_common(&p, edat);
  p.duration = 1800; p.samplingRate = 32.0; p.fHeterodyne = 0.0;
  p.sourceDeltaT = 1.0; p.dtDelayBy2 = 5; p.dtPolBy2 = 5;
  REAL4TimeSeries *g0 = XLALGeneratePulsarSignal(&p);
  if (!g0) { fprintf(stderr, "Generate(fHet=0) failed\n"); return 6; }
  dump(argv[4], g0);

  /* ----- generate with heterodyne ----- */
  fill_common(&p, edat);
  p.duration = 1800; p.samplingRate = 32.0; p.fHeterodyne = 12.0;
  p.sourceDeltaT = 1.0; p.dtDelayBy2 = 5; p.dtPolBy2 = 5;
  REAL4TimeSeries *gh = XLALGeneratePulsarSignal(&p);
  if (!gh) { fprintf(stderr, "Generate(fHet) failed\n"); return 7; }
  dump(argv[5], gh);

  fprintf(stderr, "OK\n");
  return 0;
}
