package neureka.devices.host;

import neureka.backend.api.BackendExtension;
import neureka.backend.api.ini.BackendLoader;
import neureka.backend.api.ini.ReceiveForDevice;
import neureka.backend.main.algorithms.*;
import neureka.backend.main.implementations.broadcast.*;
import neureka.backend.main.implementations.convolution.CPUConvolution;
import neureka.backend.main.implementations.elementwise.*;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.backend.main.implementations.scalar.CPUScalarFunction;
import neureka.backend.main.operations.functions.*;
import neureka.backend.main.operations.linear.CPUMatMul;
import neureka.backend.main.operations.linear.Convolution;
import neureka.backend.main.operations.linear.MatMul;
import neureka.backend.main.operations.linear.XConvLeft;
import neureka.backend.main.operations.operator.Addition;
import neureka.backend.main.operations.operator.Multiplication;
import neureka.backend.main.operations.operator.Power;
import neureka.backend.main.operations.operator.Subtraction;

public class CPUContext implements BackendExtension
{
    @Override
    public DeviceOption find(String searchKey) {
        if ( searchKey.equalsIgnoreCase("cpu")  ) new DeviceOption( CPU.get(), 1f );
        if ( searchKey.equalsIgnoreCase("jvm")  ) new DeviceOption( CPU.get(), 1f );
        if ( searchKey.equalsIgnoreCase("java") ) new DeviceOption( CPU.get(), 1f );
        return new DeviceOption( CPU.get(), 0f );
    }

    @Override
    public void dispose() { CPU.get().dispose(); }

    @Override
    public BackendLoader getLoader() {
        return registry -> _load( registry.forDevice(CPU.class) );
    }

    private void _load( ReceiveForDevice<CPU> receive )
    {
        receive.forOperation( Power.class )
                .set( Scalarization.class, context -> new CPUScalaBroadcastPower() )
                .set( Broadcast.class,     context -> new CPUBroadcastPower() )
                .set( BiElementWise.class, context -> new CPUBiElementWisePower() );

        receive.forOperation( Addition.class )
                .set( Scalarization.class, context -> new CPUScalarBroadcastAddition() )
                .set( Broadcast.class,     context -> new CPUBroadcastAddition() )
                .set( BiElementWise.class, context -> new CPUBiElementWiseAddition() );

        receive.forOperation( Subtraction.class )
                .set( Scalarization.class, context -> new CPUScalarBroadcastSubtraction() )
                .set( Broadcast.class,     context -> new CPUBroadcastSubtraction() )
                .set( BiElementWise.class, context -> new CPUBiElementWiseSubtraction() );

        receive.forOperation( Multiplication.class )
                .set( Scalarization.class, context -> new CPUScalarBroadcastMultiplication() )
                .set( Broadcast.class,     context -> new CPUBroadcastMultiplication() )
                .set( BiElementWise.class, context -> new CPUBiElementWiseMultiplication() );

        receive.forOperation( Convolution.class )
               .set( NDConvolution.class, context -> new CPUConvolution() );

        receive.forOperation( XConvLeft.class )
                .set( NDConvolution.class, context -> new CPUConvolution() );

        receive.forOperation( MatMul.class )
               .set( MatMulAlgorithm.class, context -> new CPUMatMul() );

        receive.forOperation( Absolute.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.ABSOLUTE) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.ABSOLUTE) );
        receive.forOperation( Cosinus.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.COSINUS) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.COSINUS) );
        receive.forOperation( GaSU.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.GASU) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.GASU) );
        receive.forOperation( GaTU.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.GATU) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.GATU) );
        receive.forOperation( Gaussian.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.GAUSSIAN) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.GAUSSIAN) );
        receive.forOperation( GaussianFast.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.GAUSSIAN_FAST) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.GAUSSIAN_FAST) );
        receive.forOperation( GeLU.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.GELU) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.GELU) );
        receive.forOperation( Identity.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.IDENTITY) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.IDENTITY) );
        receive.forOperation( Logarithm.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.LOGARITHM) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.LOGARITHM) );
        receive.forOperation( Quadratic.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.QUADRATIC) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.QUADRATIC) );
        receive.forOperation( ReLU.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.RELU) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.RELU) );
        receive.forOperation( SeLU.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.SELU) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.SELU) );
        receive.forOperation( Sigmoid.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.SIGMOID) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.SIGMOID) );
        receive.forOperation( SiLU.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.SILU) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.SILU) );
        receive.forOperation( Sinus.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.SINUS) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.SINUS) );
        receive.forOperation( Softplus.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.SOFTPLUS) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.SOFTPLUS) );
        receive.forOperation( Softsign.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.SOFTSIGN) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.SOFTSIGN) );
        receive.forOperation( Tanh.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.TANH) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.TANH) );
        receive.forOperation( TanhFast.class )
                .set( Activation.class, context -> new CPUElementwiseFunction( ScalarFun.TANH_FAST) )
                .set( ScalarActivation.class, context -> new CPUScalarFunction(ScalarFun.TANH_FAST) );
    }

}
