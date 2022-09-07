package neureka.backend.main.operations.function.internal;

public interface ActivationFun
{
    CPUAbsolute     ABSOLUTE      = new CPUAbsolute();
    CPUCosinus      COSINUS       = new CPUCosinus();
    CPUGaSU         GASU          = new CPUGaSU();
    CPUGaTU         GATU          = new CPUGaTU();
    CPUGaussian     GAUSSIAN      = new CPUGaussian();
    CPUGaussianFast GAUSSIAN_FAST = new CPUGaussianFast();
    CPUGeLU         GELU          = new CPUGeLU();
    CPUIdentity     IDENTITY      = new CPUIdentity();
    CPULogarithm    LOGARITHM     = new CPULogarithm();
    CPUQuadratic    QUADRATIC     = new CPUQuadratic();
    CPUReLU         RELU          = new CPUReLU();
    CPUSeLU         SELU          = new CPUSeLU();
    CPUSigmoid      SIGMOID       = new CPUSigmoid();
    CPUSiLU         SILU          = new CPUSiLU();
    CPUSinus        SINUS         = new CPUSinus();
    CPUSoftplus     SOFTPLUS      = new CPUSoftplus();
    CPUSoftsign     SOFTSIGN      = new CPUSoftsign();
    CPUTanh         TANH          = new CPUTanh();
    CPUTanhFast     TANH_FAST     = new CPUTanhFast();

    String id();

    default double calculate(double input, boolean derive) {
        if ( !derive )
            return activate( input );
        else
            return derive( input ) ;
    }

    String activationCode();

    String derivationCode();

    double activate(double x);

    double derive(double x);

    default float activate(float x) { return (float) activate( (double) x ); }

    default float derive(float x) { return (float) derive( (double) x ); }

    default int activate(int x) { return (int) Math.round( activate( (double) x ) ); }

    default int derive(int x) { return (int) Math.round( derive( (double) x ) ); }

    default long activate(long x) { return Math.round( activate( (double) x ) ); }

    default long derive(long x) { return Math.round( derive( (double) x ) ); }

    default byte activate(byte x) { return (byte) Math.round( activate( (double) x ) ); }

    default byte derive(byte x) { return (byte) Math.round( derive( (double) x ) ); }

    default short activate(short x) { return (short) Math.round( activate( (double) x ) ); }

    default short derive(short x) { return (short) Math.round( derive( (double) x ) ); }

    default boolean activate(boolean x) { return Math.round( activate( x ? 1 : 0 ) ) != 0; } // Some default behaviors, it might make sense to override this for some activations.

    default boolean derive(boolean x) { return Math.round( derive( x ? 1 : 0 ) ) != 0; } // Some default behaviors, it might make sense to override this for some activations.

    default char activate(char x) { return (char) Math.round( activate( (int) x ) ); } // Some default behaviors, it might make sense to override this for some activations.

    default char derive(char x) { return (char) Math.round( derive( (int) x ) ); } // Some default behaviors, it might make sense to override this for some activations.

    default Object activate(Object x) { throw new IllegalStateException("Not implemented for operation "+getClass().getSimpleName()); }

    default Object derive(Object x) { throw new IllegalStateException("Not implemented for operation "+getClass().getSimpleName()); }


}
