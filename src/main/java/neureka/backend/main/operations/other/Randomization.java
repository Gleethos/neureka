package neureka.backend.main.operations.other;

import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.algorithms.FallbackAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.ElementwiseAlgorithm;
import neureka.backend.main.implementations.elementwise.CLRandomization;
import neureka.backend.main.implementations.elementwise.CPURandomization;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;
import neureka.devices.host.CPU;
import neureka.devices.opencl.OpenCLDevice;
import neureka.dtype.NumericType;

import java.util.Arrays;

/**
 *  This {@link neureka.backend.api.Operation} takes an optional user seed,
 *  the shape of its input tensor, and
 *  the indices of individual elements within said tensor to generate
 *  floats or doubles with a gaussian distribution where the mean
 *  is 0 and the standard deviation is 1.
 *  This operation is very fast because it generates numbers in parallel unlike
 *  the JDKs random number generator class {@link java.util.Random}.
 */
public class Randomization extends AbstractOperation
{
    public Randomization()
    {
        super(
            new OperationBuilder()
                .identifier(       "random"   )
                .operator(         "rand"     )
                .arity(            1          )
                .isOperator(       true       )
                .isIndexer(        false      )
                .isDifferentiable( false      )
                .isInline(         true       )
        );

        setAlgorithm(
            new ElementwiseAlgorithm()
                .setIsSuitableFor(
                    call -> call.validate()
                            .allNotNull( t ->
                                t.getDataType().typeClassImplements(NumericType.class)
                                    ||
                                t.itemType() == Character.class
                                    ||
                                t.itemType() == Boolean.class
                            )
                            .basicSuitability()
                )
                .setAutogradModeFor( call -> AutoDiffMode.NOT_SUPPORTED)
                .setExecution( (caller, call) -> Result.of(AbstractDeviceAlgorithm.executeFor(caller, call, AbstractDeviceAlgorithm::executeDeviceAlgorithm)).withAutoDiff( FallbackAlgorithm::ADAction ))
                .setCallPreparation( call ->
                {
                    if ( call.input( 0 ) == null )
                        call = call.withInputAt( 0, call.input( 1 ) );

                    call.input( 0 ).mut().incrementVersion(call);

                    int hash = Arrays.hashCode( call.input( 0 ).getNDConf().shape() );
                    Arg.Seed seed = call.get(Arg.Seed.class);
                    if ( seed != null ) seed = Arg.Seed.of( CPURandomization.initialScramble(seed.get() + hash) );
                    else seed = Arg.Seed.of( CPURandomization.initialScramble(hash) );

                    return call.withArgs(seed);
                })
                .buildFunAlgorithm()
        );

    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }

}
