package neureka.backend.main.algorithms;

import neureka.Tsr;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.ADActionSupplier;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.AbstractFunDeviceAlgorithm;
import neureka.backend.main.internal.RecursiveExecutor;
import neureka.devices.Device;
import neureka.dtype.NumericType;
import neureka.ndim.NDimensional;

public final class BiElementWise extends AbstractFunDeviceAlgorithm<BiElementWise>
{
    public BiElementWise(RecursiveExecutor finalExecutor ) {
        super("elementwise");
        setIsSuitableFor(
            call -> call
                    .validate()
                    .allNotNullHaveSame(NDimensional::size)
                    .allNotNullHaveSame(NDimensional::shape)
                    .allNotNull( t -> t.getDataType().typeClassImplements( NumericType.class ) )
                    .basicSuitability()
        );
        setAutogradModeFor( call -> AutoDiffMode.FORWARD_AND_BACKWARD );
        setExecution( (outerCaller, outerCall) ->
                Result.of(AbstractDeviceAlgorithm.executeFor(
                        outerCaller, outerCall,
                        innerCall -> AbstractDeviceAlgorithm.executeDeviceAlgorithm( innerCall )
                ))
        );
        setCallPreparation(
            call -> {
                Device<Object> device = (Device<Object>) call.getDevice();
                if ( call.input( 0 ) == null ) // Creating a new tensor:
                {
                    int[] outShape = call.input( 1 ).getNDConf().shape();

                    Class<Object> type = (Class<Object>) call.input(  1 ).getItemType();
                    Tsr<Object> output = Tsr.of( type ).withShape( outShape ).all( 0.0 ).mut().setIsIntermediate( true );
                    output.mut().setIsVirtual( false );
                    try {
                        device.store( output );
                    } catch( Exception e ) {
                        e.printStackTrace();
                    }
                    call = call.withInputAt( 0, output );
                }
                return call;
            }
        );
    }

}