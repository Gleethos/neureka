package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.calculus.CalcUtil;
import neureka.devices.Device;
import neureka.dtype.NumericType;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

public class Activation extends AbstractFunctionalAlgorithm< Activation >
{

    public Activation() {
        super("activation");
        setIsSuitableFor(
                call -> call.validate()
                            .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                            .estimation()
        );
        setCanPerformBackwardADFor( call -> true );
        setCanPerformForwardADFor(
                        call -> call
                                .validate()
                                .all( ( first, second ) -> first.shape().equals(second.shape()) )
                                .isValid()
                );
        setOrchestration( CalcUtil::defaultRecursiveExecution);
        setInstantiateNewTensorsForExecutionIn(
                        call -> {
                            Tsr[] tsrs = call.getTensors();
                            Device device = call.getDevice();
                            if ( tsrs[ 0 ] == null ) // Creating a new tensor:
                            {
                                int[] shp = tsrs[ 1 ].getNDConf().shape();
                                Tsr output = Tsr.of( shp, 0.0 );
                                output.setIsVirtual( false );
                                try {
                                    device.store(output);
                                } catch( Exception e ) {
                                    e.printStackTrace();
                                }
                                tsrs[ 0 ] = output;
                            }
                            return call;
                        }
        );
    }

    public String getKernelSource() {
        return Neureka.get().utility().readResource("kernels/activation_template.cl");
    }

    @Contract(pure = true)
    public static void activate(
            Tsr t0_drn, Tsr t1_src,
            int i, int end,
            Operation.TertiaryNDIConsumer operation
    ) {
        NDIterator t0Idx = NDIterator.of( t0_drn );
        NDIterator t1Idx = NDIterator.of( t1_src );
        t0Idx.set( t0_drn.IndicesOfIndex( i ) );
        t1Idx.set( t0_drn.IndicesOfIndex( i ) );
        double[] t0_value = (double[]) t0_drn.getData();
        while ( i < end ) { // increment on drain accordingly:
            //setInto _value in drn:
            t0_value[t0Idx.i()] = operation.execute(null, t1Idx, null);
            //increment on drain:
            t0Idx.increment();
            t1Idx.increment();
            i++;
        }
    }


    @Contract(pure = true)
    public static void activate(
            Tsr t0_drn,
            int i, int end,
            Operation.TertiaryNDAConsumer operation
    ) {
        NDConfiguration ndc0 = t0_drn.getNDConf();
        int[] t0Shp = ndc0.shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
        int rank = t0Shp.length;
        int[] t0Idx = ndc0.indicesOfIndex( i );
        int[] t1Idx = new int[ rank ];
        double[] t0_value = (double[]) t0_drn.getData();
        while ( i < end ) {//increment on drain accordingly:
            System.arraycopy(t0Idx, 0, t1Idx, 0, rank);
            //setInto _value in drn:
            t0_value[ ndc0.indexOfIndices(t0Idx) ] = operation.execute( t0Idx, t1Idx, null );
            //increment on drain:
            NDConfiguration.Utility.increment( t0Idx, t0Shp );
            i++;
        }
    }


}
