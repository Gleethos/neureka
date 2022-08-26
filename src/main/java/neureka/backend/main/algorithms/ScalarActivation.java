package neureka.backend.main.algorithms;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.algorithms.internal.FunTuple;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.NumericType;
import org.jetbrains.annotations.Contract;

public class ScalarActivation extends AbstractFunDeviceAlgorithm<ScalarActivation>
{
    public ScalarActivation(FunTuple<Fun.F64ToF64> funs) {
        super("scalar activation");
        setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD );
        setIsSuitableFor( call ->
            call.validate()
                .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                .tensors( tensors ->  {
                    if ( tensors.length != 2 ) return false;
                    if ( !tensors[1].isVirtual() ) return false;
                    if ( tensors[0] != null && !tensors[0].isVirtual() ) return false;
                    return tensors[0] == null && tensors[1] != null || tensors[0].shape().equals(tensors[1].shape());
                })
                .suitabilityIfValid( SuitabilityPredicate.EXCELLENT )
        );
        setCallPreparation(
            call -> {
                Device<Number> device = call.getDeviceFor(Number.class);
                assert call.input( 0 ) == null;  // Creating a new tensor:
                int[] outShape = call.input( 1 ).getNDConf().shape();
                Class<Object> type = (Class<Object>) call.input( 1 ).getItemType();
                Tsr output = Tsr.of( type, outShape, 0.0 ).getUnsafe().setIsIntermediate( true );
                try {
                    device.store( output );
                } catch( Exception e ) {
                    e.printStackTrace();
                }
                return call.withInputAt( 0, output );
            }
        );
        setImplementationFor(
            OpenCLDevice.class,
            call -> {
                Number value =  funs.getFor( call ).invoke(call.input( Number.class, 1 ).item(0).doubleValue());
                Tsr<Number> out = call.input( Number.class, 0 );
                out.getUnsafe().setDataAt(0, value);
                return call.input(0);
            }
        );
    }

    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation( 2, call -> 1, ScalarActivation::_workloadFor );
    }

    @Contract(pure = true)
    private static CPU.RangeWorkload _workloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> functions
    ) {
        return (i, end) -> {
                    double      in  = call.input( Number.class, 1 ).item(0).doubleValue();
                    Tsr<Number> out = call.input( Number.class, 0 );
                    Number result =  functions.get(Fun.F64ToF64.class).get( call.get( Arg.DerivIdx.class ) ).invoke(in);
                    out.getUnsafe().setDataAt(0, result);
                };
    }

}