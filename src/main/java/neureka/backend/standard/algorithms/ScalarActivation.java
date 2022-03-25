package neureka.backend.standard.algorithms;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.backend.standard.algorithms.internal.Fun;
import neureka.backend.standard.algorithms.internal.FunArray;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.NumericType;
import org.jetbrains.annotations.Contract;

public class ScalarActivation extends AbstractFunctionalAlgorithm<ScalarActivation>
{
    public ScalarActivation(FunArray<Fun.F64ToF64> funs) {
        super("scalar activation");
        setCanPerformBackwardADFor( call -> true );
        setCanPerformForwardADFor( call -> true );
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
                Class<Object> type = (Class<Object>) call.input( 1 ).getValueClass();
                Tsr output = Tsr.of( type, outShape, 0.0 ).getUnsafe().setIsIntermediate( true );
                try {
                    device.store( output );
                } catch( Exception e ) {
                    e.printStackTrace();
                }
                call.setInput( 0, output );
                return call;
            }
        );
        setImplementationFor(
            OpenCLDevice.class,
            call -> {
                Number value =  funs.get( call.getValOf( Arg.DerivIdx.class ) ).invoke(call.getTsrOfType( Number.class, 1 ).getValueAt(0).doubleValue());
                Tsr<Number> out = call.getTsrOfType( Number.class, 0 );
                out.setDataAt(0, value);
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
        CPU.RangeWorkload workload = (i,end) -> {
            Number value =  functions.get(Fun.F64ToF64.class).get( call.getValOf( Arg.DerivIdx.class ) ).invoke(call.getTsrOfType( Number.class, 1 ).getValueAt(0).doubleValue());
            Tsr<Number> out = call.getTsrOfType( Number.class, 0 );
            out.setDataAt(0, value);
        };
        return workload;
    }

}