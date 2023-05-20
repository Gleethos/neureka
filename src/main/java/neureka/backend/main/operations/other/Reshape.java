package neureka.backend.main.operations.other;

import neureka.Tensor;
import neureka.backend.api.Algorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.framing.Relation;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.ndim.NDConstructor;
import neureka.ndim.config.NDConfiguration;

public class Reshape extends AbstractOperation
{
    public Reshape()
    {
        super(
            new OperationBuilder()
                .identifier(       "reshape"  )
                .operator(         "reshape"  )
                .arity(            1          )
                .isOperator(       false      )
                .isIndexer(        false      )
                .isDifferentiable( true       )
                .isInline(         false      )
        );
        setAlgorithm(
            Algorithm
            .withName( "reshape" )
            .setIsSuitableFor( call -> SuitabilityPredicate.GOOD )
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution(
                ( caller, call ) ->
                {
                    Tensor<?>[] inputs = AbstractDeviceAlgorithm.flatten(caller, call).inputs();
                    Tensor<Object> input = (Tensor<Object>) inputs[0];

                    int[] foundShape = call.getValOf( Arg.Shape.class );

                    if ( foundShape == null )
                        throw new IllegalArgumentException("Shape argument is missing!");

                    int[] shape = _resolveNewShape(input.size(), foundShape);

                    Tensor reshaped = Tensor.of(
                                    input.getDataType(),
                                    NDConstructor.of( shape ),
                                    input.mut().getData()
                                );

                    reshaped.set( Relation.newChildToParent( input ) );
                    Relation parent = input.find( Relation.class ).orElseGet(Relation::newParentToChildren);
                    parent.addChild( reshaped );
                    input.set( parent );

                    if ( input.isOutsourced() )
                        input.getDevice().store( reshaped );

                    NDConfiguration originalConfig = input.getNDConf();

                    return Result.of(reshaped.mut().setIsIntermediate(true))
                            .withADAction( target -> {
                                Tensor<Object> error = (Tensor<Object>) target.error();
                                return Tensor.of(
                                        error.getDataType(),
                                        NDConstructor.of( originalConfig ),
                                        error.mut().getData()
                                    );
                            });
                }
            )
            .buildFunAlgorithm()
        );
    }

    /**
     *   If the provided shape array contains a -1 as one of its elements,
     *   then this method will resolve the -1 to the correct value
     *   which results in a shape array which is compatible with the provided size,
     *   meaning that when we multiply all the elements of the resolved shape array
     *   we will get the provided size.
     *
     * @param size The size which the resolved shape array should be compatible with.
     * @param shape The shape array which may contain a -1.
     * @return The resolved shape array.
     */
    private static int[] _resolveNewShape( int size, int[] shape )
    {
        int[] resolvedShape = new int[ shape.length ];
        int minusOneIndex = -1;
        int minusOneCount = 0;
        int product = 1;
        for ( int i = 0; i < shape.length; i++ )
        {
            if ( shape[ i ] == -1 )
            {
                minusOneIndex = i;
                minusOneCount++;
            }
            else
            {
                resolvedShape[ i ] = shape[ i ];
                product *= shape[ i ];
            }
        }
        if ( minusOneCount > 1 )
            throw new IllegalArgumentException("The shape array contains more than one -1!");
        if ( minusOneCount == 1 )
            resolvedShape[ minusOneIndex ] = size / product;
        return resolvedShape;
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src )
    {
        return src[ 0 ].call( inputs, j );
    }
}
