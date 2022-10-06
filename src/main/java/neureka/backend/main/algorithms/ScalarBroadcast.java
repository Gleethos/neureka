package neureka.backend.main.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.calculus.args.Arg;
import neureka.devices.Device;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.NumericType;

public class ScalarBroadcast extends AbstractFunDeviceAlgorithm<ScalarBroadcast>
{
    public ScalarBroadcast(ScalarFun fun) {
        super("scalar broadcast");
        setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD );
        setIsSuitableFor( call ->
                call.validate()
                        .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                        //.first( Objects::isNull )
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
                    Tsr output = Tsr.of( type, outShape, 0.0 ).getMut().setIsIntermediate( true );
                    output.getMut().setIsVirtual( false );
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


    public static String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/scalar_broadcast.cl");
    }

}