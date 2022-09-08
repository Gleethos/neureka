package neureka.backend.main.implementations.fun.api;

import neureka.backend.main.implementations.fun.*;

public interface ScalarFun
{
    ScalarAbsolute ABSOLUTE          = new ScalarAbsolute();
    ScalarCosinus COSINUS            = new ScalarCosinus();
    ScalarGaSU GASU                  = new ScalarGaSU();
    ScalarGaTU GATU                  = new ScalarGaTU();
    ScalarGaussian GAUSSIAN          = new ScalarGaussian();
    ScalarGaussianFast GAUSSIAN_FAST = new ScalarGaussianFast();
    ScalarGeLU GELU                  = new ScalarGeLU();
    ScalarIdentity IDENTITY          = new ScalarIdentity();
    ScalarLogarithm LOGARITHM        = new ScalarLogarithm();
    ScalarQuadratic QUADRATIC        = new ScalarQuadratic();
    ScalarReLU RELU                  = new ScalarReLU();
    ScalarSeLU SELU                  = new ScalarSeLU();
    ScalarSigmoid SIGMOID            = new ScalarSigmoid();
    ScalarSiLU SILU                  = new ScalarSiLU();
    ScalarSinus SINUS                = new ScalarSinus();
    ScalarSoftplus SOFTPLUS          = new ScalarSoftplus();
    ScalarSoftsign SOFTSIGN          = new ScalarSoftsign();
    ScalarTanh TANH                  = new ScalarTanh();
    ScalarTanhFast TANH_FAST         = new ScalarTanhFast();

    String id();

    String activationCode();

    String derivationCode();

    default double calculate(double input, boolean derive) {
        if ( !derive )
            return getActivation().invoke( input );
        else
            return getDerivative().invoke( input ) ;
    }

    CPUFun getActivation();

    CPUFun getDerivative();

}
