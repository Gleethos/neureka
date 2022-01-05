package neureka.backend.standard.algorithms;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.Operation;
import neureka.backend.api.algorithms.AbstractFunctionalAlgorithm;
import neureka.calculus.CalcUtil;
import neureka.devices.Device;
import neureka.dtype.NumericType;
import neureka.ndim.iterators.NDIterator;
import org.jetbrains.annotations.Contract;

/**
 *  This is lambda based {@link neureka.backend.api.Algorithm} implementation
 *  providing some basic functionality for implementing custom
 *  activation functions.
 */
public class Activation extends AbstractFunctionalAlgorithm<Activation>
{

    public Activation() {
        super("activation");
        setIsSuitableFor(
                call -> call.validate()
                            .allNotNull( t -> t.getDataType().typeClassImplements(NumericType.class) )
                            .basicSuitability()
        );
        setCanPerformBackwardADFor( call -> true );
        setCanPerformForwardADFor(
                        call -> call
                                .validate()
                                .all( ( first, second ) -> first.shape().equals(second.shape()) )
                                .isValid()
                );
        setExecutionDispatcher( CalcUtil::defaultRecursiveExecution);
        setCallPreparation(
                        call -> {
                            Tsr<?>[] tensors = call.getTensors();
                            Device<Number> device = call.getDeviceFor(Number.class);
                            if ( tensors[ 0 ] == null ) // Creating a new tensor:
                            {
                                int[] shp = tensors[ 1 ].getNDConf().shape();
                                Tsr<Double> output = Tsr.of( shp, 0.0 );
                                output.setIsVirtual( false );
                                try {
                                    device.store( output );
                                } catch( Exception e ) {
                                    e.printStackTrace();
                                }
                                tensors[ 0 ] = output;
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
            Tsr<?> t0_drn, Tsr<?> t1_src,
            int i, int end,
            Operation.TertiaryF64NDFun operation
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


}
