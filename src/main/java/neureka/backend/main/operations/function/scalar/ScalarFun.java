package neureka.backend.main.operations.function.scalar;

import neureka.backend.api.ImplementationFor;
import neureka.devices.host.CPU;

public interface ScalarFun
{
    ScalarAbsolute ABSOLUTE      = new ScalarAbsolute();
    ScalarCosinus COSINUS       = new ScalarCosinus();
    ScalarGaSU GASU          = new ScalarGaSU();
    ScalarGaTU GATU          = new ScalarGaTU();
    ScalarGaussian GAUSSIAN      = new ScalarGaussian();
    ScalarGaussianFast GAUSSIAN_FAST = new ScalarGaussianFast();
    ScalarGeLU GELU          = new ScalarGeLU();
    ScalarIdentity IDENTITY      = new ScalarIdentity();
    ScalarLogarithm LOGARITHM     = new ScalarLogarithm();
    ScalarQuadratic QUADRATIC     = new ScalarQuadratic();
    ScalarReLU RELU          = new ScalarReLU();
    ScalarSeLU SELU          = new ScalarSeLU();
    ScalarSigmoid SIGMOID       = new ScalarSigmoid();
    ScalarSiLU SILU          = new ScalarSiLU();
    ScalarSinus SINUS         = new ScalarSinus();
    ScalarSoftplus SOFTPLUS      = new ScalarSoftplus();
    ScalarSoftsign SOFTSIGN      = new ScalarSoftsign();
    ScalarTanh TANH          = new ScalarTanh();
    ScalarTanhFast TANH_FAST     = new ScalarTanhFast();

    default ImplementationFor<CPU> elementwise() {
        return new CPUElementwiseActivation(this);
    }
    
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
