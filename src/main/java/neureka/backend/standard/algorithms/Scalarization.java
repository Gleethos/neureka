package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Fun;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.NumericType;
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


    public static Functions.Builder<Fun> implementationForCPU() {
        return Functions.implementation(
                (call, pairs) ->
                        call.getDevice()
                                .getExecutor()
                                .threaded(
                                        call.getTsrOfType( Number.class, 0 ).size(),
                                        _newWorkloadFor( call, pairs )
                                )
        );
    }
    @Contract(pure = true)
    public static CPU.RangeWorkload _newWorkloadFor(
        ExecutionCall<CPU> call,
        Functions<Fun> functions
    ) {
        double value = call.getTsrOfType( Number.class, 2 ).getDataAs( double[].class )[ 0 ];
        Tsr<?> t0_drn = call.getTensors()[1];
        Tsr<?> src    = call.getTensors()[2];

        Fun.F64F64ToF64 operation = functions.get(Fun.F64F64ToF64.class).get(call.getDerivativeIndex() );

        double[] t0_value = t0_drn.getDataAs(double[].class);
        double[] t1_value = src.getDataAs(double[].class);

        return ( i, end ) -> {
            NDIterator t0Idx = NDIterator.of(t0_drn);
            NDIterator srcIdx = NDIterator.of(src);
            t0Idx.set(t0_drn.IndicesOfIndex(i));
            srcIdx.set(src.IndicesOfIndex(i));
            while ( i < end ) // increment on drain accordingly:
            {
                // setInto _value in drn:
                t0_value[t0Idx.i()] = operation.invoke(t1_value[srcIdx.i()], value);
                // increment on drain:
                t0Idx.increment();
                srcIdx.increment();
                //NDConfiguration.Utility.increment(t0Idx, t0Shp);
                i++;
            }
        };
    }


    @Contract(pure = true)
    public static void scalarize (
            Tsr<?> t0_drn, Tsr<?> src,
            int i, int end,
            Operation.PrimaryF64NDFun operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator srcIdx = NDIterator.of( src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        srcIdx.set( src.IndicesOfIndex( i ) );
        double[] t0_value = t0_drn.getDataAs( double[].class );
        while ( i < end ) // increment on drain accordingly:
        {
            // setInto _value in drn:
            t0_value[ t0Idx.i() ] = operation.execute( srcIdx );
            // increment on drain:
            t0Idx.increment();
            srcIdx.increment();
            //NDConfiguration.Utility.increment(t0Idx, t0Shp);
            i++;
        }
    }

}