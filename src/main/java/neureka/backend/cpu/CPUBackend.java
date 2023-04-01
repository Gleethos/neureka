package neureka.backend.cpu;

import neureka.backend.api.BackendExtension;
import neureka.backend.api.ini.BackendLoader;
import neureka.backend.api.ini.ReceiveForDevice;
import neureka.backend.main.algorithms.*;
import neureka.backend.main.implementations.broadcast.*;
import neureka.backend.main.implementations.convolution.CPUConvolution;
import neureka.backend.main.implementations.elementwise.*;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.backend.main.implementations.linear.CPUDot;
import neureka.backend.main.implementations.matmul.CPUMatMul;
import neureka.backend.main.implementations.scalar.CPUScalarFunction;
import neureka.backend.main.operations.functions.*;
import neureka.backend.main.operations.linear.*;
import neureka.backend.main.operations.operator.*;
import neureka.backend.main.operations.other.AssignLeft;
import neureka.backend.main.operations.other.Randomization;
import neureka.backend.main.operations.other.Sum;
import neureka.backend.main.operations.other.internal.CPUSum;
import neureka.devices.host.CPU;

/**
 *  This class loads the CPU operations into the Neureka library context.
 */
public class CPUBackend implements BackendExtension
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
    public BackendLoader getLoader() { return registry -> _load( registry.forDevice(CPU.class) ); }

    private void _load( ReceiveForDevice<CPU> receive )
    {
        receive.forOperation( Power.class )
                .set( BiScalarBroadcast.class, context -> new CPUScalaBroadcastPower() )
                .set( Broadcast.class,     context -> new CPUBroadcastPower() )
                .set( BiElementwise.class, context -> new CPUBiElementWisePower() );

        receive.forOperation( Addition.class )
                .set( BiScalarBroadcast.class, context -> new CPUScalarBroadcastAddition() )
                .set( Broadcast.class,     context -> new CPUBroadcastAddition() )
                .set( BiElementwise.class, context -> new CPUBiElementWiseAddition() );

        receive.forOperation( Subtraction.class )
                .set( BiScalarBroadcast.class, context -> new CPUScalarBroadcastSubtraction() )
                .set( Broadcast.class,     context -> new CPUBroadcastSubtraction() )
                .set( BiElementwise.class, context -> new CPUBiElementWiseSubtraction() );

        receive.forOperation( Multiplication.class )
                .set( BiScalarBroadcast.class, context -> new CPUScalarBroadcastMultiplication() )
                .set( Broadcast.class,     context -> new CPUBroadcastMultiplication() )
                .set( BiElementwise.class, context -> new CPUBiElementWiseMultiplication() );

        receive.forOperation( Division.class )
                .set( BiScalarBroadcast.class, context -> new CPUScalarBroadcastDivision() )
                .set( Broadcast.class,     context -> new CPUBroadcastDivision() )
                .set( BiElementwise.class, context -> new CPUBiElementWiseDivision() );

        receive.forOperation( Modulo.class )
                .set( BiScalarBroadcast.class, context -> new CPUScalarBroadcastModulo() )
                .set( Broadcast.class,     context -> new CPUBroadcastModulo() )
                .set( BiElementwise.class, context -> new CPUBiElementWiseModulo() );

        receive.forOperation( AssignLeft.class )
                .set( BiScalarBroadcast.class, context -> new CPUScalarBroadcastIdentity() )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseAssignFun() );

        receive.forOperation( Convolution.class )
               .set( NDConvolution.class, context -> new CPUConvolution() );
        receive.forOperation( XConvLeft.class )
                .set( NDConvolution.class, context -> new CPUConvolution() );
        receive.forOperation( XConvRight.class )
                .set( NDConvolution.class, context -> new CPUConvolution() );

        receive.forOperation( MatMul.class )
                .set( MatMulAlgorithm.class, context -> new CPUMatMul() );

        receive.forOperation( DotProduct.class )
                .set( DotProductAlgorithm.class, context -> new CPUDot() );

        receive.forOperation( Sum.class )
                .set( SumAlgorithm.class, context -> new CPUSum() );

        receive.forOperation( Randomization.class )
                .set( ElementwiseAlgorithm.class, context -> new CPURandomization() );

        receive.forOperation( Absolute.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.ABSOLUTE) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.ABSOLUTE) );
        receive.forOperation( Cosinus.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.COSINUS) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.COSINUS) );
        receive.forOperation( GaSU.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.GASU) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.GASU) );
        receive.forOperation( GaTU.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.GATU) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.GATU) );
        receive.forOperation( Gaussian.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.GAUSSIAN) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.GAUSSIAN) );
        receive.forOperation( GaussianFast.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.GAUSSIAN_FAST) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.GAUSSIAN_FAST) );
        receive.forOperation( GeLU.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.GELU) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.GELU) );
        receive.forOperation( Identity.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseAssignFun() )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.IDENTITY) );
        receive.forOperation( Logarithm.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.LOGARITHM) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.LOGARITHM) );
        receive.forOperation( Quadratic.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.QUADRATIC) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.QUADRATIC) );
        receive.forOperation( ReLU.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.RELU) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.RELU) );
        receive.forOperation( SeLU.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.SELU) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.SELU) );
        receive.forOperation( Sigmoid.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.SIGMOID) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.SIGMOID) );
        receive.forOperation( SiLU.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.SILU) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.SILU) );
        receive.forOperation( Sinus.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.SINUS) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.SINUS) );
        receive.forOperation( Softplus.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.SOFTPLUS) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.SOFTPLUS) );
        receive.forOperation( Softsign.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.SOFTSIGN) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.SOFTSIGN) );
        receive.forOperation( Tanh.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.TANH) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.TANH) );
        receive.forOperation( TanhFast.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.TANH_FAST) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.TANH_FAST) );
        receive.forOperation( Exp.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.EXP) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.EXP) );
        receive.forOperation( Cbrt.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.CBRT) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.CBRT) );
        receive.forOperation( Log10.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.LOG10) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.LOG10) );
        receive.forOperation( Sqrt.class )
                .set( ElementwiseAlgorithm.class, context -> new CPUElementwiseFunction( ScalarFun.SQRT) )
                .set( ScalarAlgorithm.class, context -> new CPUScalarFunction(ScalarFun.SQRT) );
    }

}
