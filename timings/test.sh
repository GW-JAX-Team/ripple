DEVICE="cpu"
N_WAVEFORMS="1000"
PRECISION="float32"

ripple_time TaylorF2 --device $DEVICE --n-waveforms $N_WAVEFORMS --precision $PRECISION
ripple_time IMRPhenomD --device $DEVICE --n-waveforms $N_WAVEFORMS --precision $PRECISION
ripple_time IMRPhenomXAS --device $DEVICE --n-waveforms $N_WAVEFORMS --precision $PRECISION
ripple_time IMRPhenomPv2 --device $DEVICE --n-waveforms $N_WAVEFORMS --precision $PRECISION
