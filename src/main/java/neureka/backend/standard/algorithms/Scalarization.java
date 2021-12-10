package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.devices.Device;
import neureka.dtype.NumericType;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

public class Scalarization extends AbstractFunctionalAlgorithm< Scalarization >
{
    public Scalarization() {
        super("scalarization");
        setCanPerformBackwardADFor( call -> true );
        setCanPerformForwardADFor( call -> true );
        setIsSuitableFor( call ->
                                call.validate()
                                        .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                                        //.first( Objects::isNull )
                                        .tensors( tensors ->  {
                                            if ( tensors.length != 2 && tensors.length != 3 ) return false;
                                            int offset = ( tensors.length == 2 ? 0 : 1 );
                                            if ( tensors[1+offset].size() > 1 && !tensors[1+offset].isVirtual() ) return false;
                                            return
                                                //tensors[1+offset].shape().stream().allMatch( d -> d == 1 )
                                                //||
                                                tensors[offset].shape().equals(tensors[1+offset].shape());
                                        })
                                        .suitabilityIfValid( SuitabilityPredicate.VERY_GOOD )
        );
        setCallPreparation(
                call -> {
                    Tsr<?>[] tensors = call.getTensors();
                    Device<Number> device = call.getDeviceFor(Number.class);
                    assert tensors[ 0 ] == null;  // Creating a new tensor:

                    int[] shp = tensors[ 1 ].getNDConf().shape();
                    Tsr<Double> output = Tsr.of( shp, 0.0 );
                    output.setIsVirtual( false );
                    try {
                        device.store( output );
                    } catch( Exception e ) {
                        e.printStackTrace();
                    }
                    tensors[ 0 ] = output;

                    return call;
                }
        );
    }


    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/scalarization_template.cl");
    }


    @Contract(pure = true)
    public static void scalarize (
            Tsr<?> t0_drn,
            int i, int end,
            Operation.PrimaryNDIConsumer operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        double[] t0_value = t0_drn.getDataAs( double[].class );
        while ( i < end ) // increment on drain accordingly:
        {
            // setInto _value in drn:
            t0_value[ t0Idx.i() ] = operation.execute( t0Idx );
            // increment on drain:
            t0Idx.increment();
            //NDConfiguration.Utility.increment(t0Idx, t0Shp);
            i++;
        }
    }

    @Contract(pure = true)
    public static void scalarize (
            Tsr<?> t0_drn,
            int i, int end,
            Operation.PrimaryNDAConsumer operation
    ) {
        int[] t0Shp = t0_drn.getNDConf().shape();
        int[] t0Idx = t0_drn.IndicesOfIndex( i );
        double[] t0_value = t0_drn.getDataAs( double[].class );
        while ( i < end ) // increment on drain accordingly:
        {
            // setInto _value in drn:
            t0_value[ t0_drn.indexOfIndices(t0Idx) ] = operation.execute( t0Idx );
            // increment on drain:
            NDConfiguration.Utility.increment(t0Idx, t0Shp);
            i++;
        }
    }


}