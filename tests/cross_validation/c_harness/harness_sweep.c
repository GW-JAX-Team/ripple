/* Sweep harness: run the compiled XLAL pulsar-signal functions over many
 * parameter sets read from a CSV and dump all output time series to one binary
 * file. See harness.c for the struct ABI notes (self-verified at runtime).
 *
 * CSV columns (one set per line, whitespace separated):
 *   mode alpha delta f0 f1 f2 phi0 psi aplus across asini ecc period argp tp fhet
 * mode: 0 = XLALSimulateExactPulsarSignal, 1 = XLALGeneratePulsarSignal.
 * asini == 0 -> isolated; asini > 0 -> binary (generate mode only).
 *
 * Output binary: UINT4 n_sets, UINT4 n_samples, then for each set n_samples
 * float32 strain values (h(t) = detector strain). duration/fs are fixed below.
 *
 * Usage: ./harness_sweep earth sun params.csv out.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef double REAL8; typedef float REAL4; typedef int INT4; typedef unsigned int UINT4;
typedef short INT2; typedef unsigned short UINT2; typedef char CHAR;
enum { LALNameLength = 64 };
#define LALNumUnits 7

typedef struct { INT4 gpsSeconds, gpsNanoSeconds; } LIGOTimeGPS;
typedef struct { INT2 powerOfTen; INT2 unitNumerator[LALNumUnits]; UINT2 unitDenominatorMinusOne[LALNumUnits]; } LALUnit;
typedef struct { UINT4 length; REAL8 *data; } REAL8Vector;
typedef struct { UINT4 length; REAL4 *data; } REAL4Sequence;
typedef struct { CHAR name[LALNameLength]; LIGOTimeGPS epoch; REAL8 deltaT, f0; LALUnit sampleUnits; REAL4Sequence *data; } REAL4TimeSeries;
typedef enum { CS_HORIZON, CS_GEOGRAPHIC, COORDINATESYSTEM_EQUATORIAL, CS_ECLIPTIC, CS_GALACTIC } CoordinateSystem;
typedef struct { REAL8 longitude, latitude; CoordinateSystem system; } SkyPosition;
typedef enum { DT_ABSENT, DT_IFODIFF, DT_IFOXARM, DT_IFOYARM, DT_IFOCOMM, DT_CYLBAR } LALDetectorType;
typedef struct {
  CHAR name[LALNameLength]; CHAR prefix[3];
  REAL8 vertexLongitudeRadians, vertexLatitudeRadians;
  REAL4 vertexElevation, xArmAltitudeRadians, xArmAzimuthRadians,
        yArmAltitudeRadians, yArmAzimuthRadians, xArmMidpoint, yArmMidpoint;
} LALFrDetector;
typedef struct { REAL8 location[3]; REAL4 response[3][3]; LALDetectorType type; LALFrDetector frDetector; } LALDetector;
typedef struct tagEphemerisData EphemerisData;
typedef struct tagCOMPLEX8FrequencySeries COMPLEX8FrequencySeries;
typedef struct {
  struct { LIGOTimeGPS refTime; SkyPosition position; REAL4 psi, aPlus, aCross; REAL8 phi0, f0; REAL8Vector *spindown; } pulsar;
  struct { LIGOTimeGPS tp; REAL8 argp, asini, ecc, period; } orbit;
  REAL8 sourceDeltaT;
  const COMPLEX8FrequencySeries *transfer;
  const LALDetector *site;
  const EphemerisData *ephemerides;
  LIGOTimeGPS startTimeGPS;
  UINT4 duration; REAL8 samplingRate, fHeterodyne; UINT4 dtDelayBy2, dtPolBy2;
} PulsarSignalParams;

extern const LALDetector lalCachedDetectors[];
extern EphemerisData *XLALInitBarycenter(const char *, const char *);
extern REAL8Vector *XLALCreateREAL8Vector(UINT4);
extern void XLALDestroyREAL8Vector(REAL8Vector *);
extern REAL4TimeSeries *XLALSimulateExactPulsarSignal(const PulsarSignalParams *);
extern REAL4TimeSeries *XLALGeneratePulsarSignal(const PulsarSignalParams *);
extern void XLALDestroyREAL4TimeSeries(REAL4TimeSeries *);

#define LHODIFF 5
static const int START_GPS = 1000000000;
static const UINT4 DURATION = 1800;
static const double FS = 16.0;

int main(int argc, char **argv) {
  if (argc != 5) { fprintf(stderr, "usage: %s earth sun params.csv out.bin\n", argv[0]); return 2; }
  const LALDetector *h1 = &lalCachedDetectors[LHODIFF];
  if (fabs(h1->location[0] - (-2161414.92636)) > 1e-3) { fprintf(stderr, "ABI mismatch\n"); return 3; }
  EphemerisData *edat = XLALInitBarycenter(argv[1], argv[2]);
  if (!edat) { fprintf(stderr, "InitBarycenter failed\n"); return 4; }

  FILE *cf = fopen(argv[3], "r");
  if (!cf) { fprintf(stderr, "cannot open %s\n", argv[3]); return 5; }

  /* count rows */
  char line[1024]; UINT4 n_sets = 0;
  while (fgets(line, sizeof(line), cf)) if (line[0] != '\n' && line[0] != '#') n_sets++;
  rewind(cf);

  UINT4 n_samples = (UINT4)ceil(FS * DURATION);
  FILE *of = fopen(argv[4], "wb");
  if (!of) { fprintf(stderr, "cannot open %s for writing\n", argv[4]); return 9; }
  fwrite(&n_sets, sizeof(UINT4), 1, of);
  fwrite(&n_samples, sizeof(UINT4), 1, of);

  UINT4 done = 0;
  while (fgets(line, sizeof(line), cf)) {
    if (line[0] == '\n' || line[0] == '#') continue;
    int mode; double al, de, f0, f1, f2, phi0, psi, ap, ac, asini, ecc, per, argp, tp, fhet;
    if (sscanf(line, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
               &mode, &al, &de, &f0, &f1, &f2, &phi0, &psi, &ap, &ac,
               &asini, &ecc, &per, &argp, &tp, &fhet) != 16) {
      fprintf(stderr, "bad line: %s", line); return 6;
    }
    PulsarSignalParams p; memset(&p, 0, sizeof(p));
    p.pulsar.refTime.gpsSeconds = 0;
    p.pulsar.position.longitude = al; p.pulsar.position.latitude = de;
    p.pulsar.position.system = COORDINATESYSTEM_EQUATORIAL;
    p.pulsar.psi = psi; p.pulsar.aPlus = ap; p.pulsar.aCross = ac;
    p.pulsar.phi0 = phi0; p.pulsar.f0 = f0;
    REAL8Vector *sd = XLALCreateREAL8Vector(2); sd->data[0] = f1; sd->data[1] = f2;
    p.pulsar.spindown = sd;
    p.site = h1; p.ephemerides = edat;
    p.startTimeGPS.gpsSeconds = START_GPS;
    p.duration = DURATION; p.samplingRate = FS; p.fHeterodyne = fhet;
    if (asini > 0.0) {
      p.orbit.asini = asini; p.orbit.ecc = ecc; p.orbit.period = per;
      p.orbit.argp = argp; p.orbit.tp.gpsSeconds = (INT4)tp;
      p.orbit.tp.gpsNanoSeconds = (INT4)round((tp - (INT4)tp) * 1e9);
    }
    if (mode == 1) { p.sourceDeltaT = 1.0; p.dtDelayBy2 = 5; p.dtPolBy2 = 5; }

    REAL4TimeSeries *ts = (mode == 0) ? XLALSimulateExactPulsarSignal(&p)
                                      : XLALGeneratePulsarSignal(&p);
    if (!ts) { fprintf(stderr, "signal failed at set %u\n", done); return 7; }
    if (ts->data->length != n_samples) {
      fprintf(stderr, "len mismatch %u != %u\n", ts->data->length, n_samples); return 8;
    }
    fwrite(ts->data->data, sizeof(REAL4), n_samples, of);
    XLALDestroyREAL4TimeSeries(ts);
    XLALDestroyREAL8Vector(sd);
    if (++done % 25 == 0) fprintf(stderr, "  %u/%u\n", done, n_sets);
  }
  fclose(cf); fclose(of);
  fprintf(stderr, "OK: %u sets x %u samples\n", n_sets, n_samples);
  return 0;
}
