package neureka.backend.main.algorithms;

import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.backend.api.template.algorithms.FallbackAlgorithm;
import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.backend.main.implementations.scalar.CPUScalarBroadcastFunction;
import neureka.math.args.Arg;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.NumericType;
import neureka.ndim.NDimensional;

public class ScalarBroadcast extends AbstractFunDeviceAlgorithm<ScalarBroadcast>
{
    public ScalarBroadcast(ScalarFun fun) {
        super("scalar broadcast");
        setAutogradModeFor(
            call -> call
                    .validate().allNotNullHaveSame(NDimensional::shape)
                    .ifValid(AutoDiffMode.FORWARD_AND_BACKWARD)
                    .orElse(AutoDiffMode.BACKWARD_ONLY)
        );
        setIsSuitableFor( call ->
                call.validate()
                        .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                        .tensors( tensors ->  {
                            if ( tensors.length != 2 ) return false;
                            if ( !tensors[1].isVirtual() ) return false;
                            if ( tensors[0] != null && tensors[0].isVirtual() ) return false;
                            return tensors[0] == null && tensors[1] != null || tensors[0].shape().equals(tensors[1].shape());
                        })
                        .suitabilityIfValid( SuitabilityPredicate.VERY_GOOD )
        );
        setCallPreparation(
                call -> {
                    Device<Number> device = call.getDeviceFor(Number.class);
                    assert call.input( 0 ) == null;  // Creating a new tensor:
                    int[] outShape = call.input( 1 ).getNDConf().shape();
                    Class<Object> type = (Class<Object>) call.input( 1 ).getItemType();
                    Tsr output = Tsr.of( type, outShape, 0.0 ).mut().setIsIntermediate( true );
                    output.mut().setIsVirtual( false );
                    try {
                        device.store( output );
                    } catch( Exception e ) {
                        e.printStackTrace();
                    }
                    return call.withInputAt( 0, output );
                }
        );
        setExecution(
            (caller, call) ->
                Result.of(AbstractDeviceAlgorithm.prepareAndExecute(call,AbstractDeviceAlgorithm::executeDeviceAlgorithm))
                        .withAutoDiff( FallbackAlgorithm::ADAction )
        );

        setImplementationFor( CPU.class, new CPUScalarBroadcastFunction( fun ) );
        setImplementationFor(
            OpenCLDevice.class,
            call -> {
                int d = call.getValOf(Arg.DerivIdx.class);
                CPUFun f = d < 0 ? fun.getActivation() : fun.getDerivative();
                double value =  f.invoke( call.input( Number.class, 1 ).at(0).get().doubleValue() );
                Tsr<Number> t = call.input( Number.class, 0 );
                int gwz = t.size();
                call.getDevice()
                        .getKernel("scalar_broadcast")
                        .passAllOf(t)
                        .pass((float) value)
                        .pass(t.rank())
                        .call( gwz );

                return call.input(0);
            }
        );
    }

}