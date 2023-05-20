package neureka.backend.main.operations.other;

import neureka.Neureka;
import neureka.Tensor;
import neureka.backend.api.Algorithm;
import neureka.backend.api.AutoDiffMode;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.backend.api.fun.SuitabilityPredicate;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.framing.NDFrame;
import neureka.math.Function;
import neureka.math.args.Arg;

import java.util.*;
import java.util.stream.Collectors;

public class Cat extends AbstractOperation
{
    public Cat()
    {
        super(
            new OperationBuilder()
                .identifier(       "concat"    )
                .operator(         "concat"    )
                .arity(            -1          ) // Any number of arguments
                .isOperator(       false       )
                .isIndexer(        false       )
                .isDifferentiable( true        )
                .isInline(         false       )
        );
        setAlgorithm(
            Algorithm
            .withName("concat")
            .setIsSuitableFor( call -> {
                Integer dim = call.getValOf(Arg.Axis.class);
                Tensor<?> a = call.input(0);
                Tensor<?> b = call.input(1);
                if ( a.rank() != b.rank() ) return SuitabilityPredicate.UNSUITABLE;
                for ( int i = 0; i < a.rank(); i++ )
                    if ( i != dim && a.shape(i) != b.shape(i) )
                        return SuitabilityPredicate.UNSUITABLE;

                return SuitabilityPredicate.GOOD;
            })
            .setAutogradModeFor( call -> AutoDiffMode.BACKWARD_ONLY )
            .setExecution(
                ( caller, call ) ->
                {
                    // The dimension alongside we want to concat:
                    Integer dim = call.getValOf(Arg.Axis.class);

                    // First let's find out the shape of the concatenated result:
                    Tensor<?>[] inputs = call.inputs();
                    List<Integer> axes = Arrays.stream(inputs).map( t -> t.shape(dim) ).collect(Collectors.toList());
                    int newAxisSize = axes.stream().mapToInt( i -> i ).sum();
                    List<Integer> newShape = new ArrayList<>();
                    for ( int i = 0; i < call.input(0).rank(); i++ )
                        newShape.add( i == dim ? newAxisSize : call.input(0).shape(i) );

                    // We create the output tensor:
                    Tensor<?> c = Tensor.of( call.input(0).getItemType(), newShape, 0 );

                    // We make the axes list entries cumulative:
                    for ( int i = 0; i < axes.size(); i++ )
                        axes.set( i, ( i == 0 ? axes.get(i) : axes.get( i - 1 ) + axes.get(i) ) );

                    // Now we need to create the slices of c needed to populate c:
                    for ( int i = 0; i < inputs.length; i++ ) {
                        int start = i == 0 ? 0 : axes.get( i - 1 );
                        int end = ( axes.get( i ) - 1 );
                        Tensor<?> slice = c.slice().axis( dim ).from( start ).to( end ).detached();
                        Neureka.get().backend().getFunction().idy().execute( slice, call.input( i ) );
                    }
                    c.mut().setIsIntermediate(true);
                    try {
                        _catFrames( inputs, c, dim );
                    } catch ( Exception e ) {
                        e.printStackTrace();
                        // Framing is not that important, a result however is!
                        // So an exception in the frame concatenation is not fatal!
                    }
                    return
                        Result.of(c)
                            .withADAction( target -> {
                                int i = target.inputIndex();
                                int start = i == 0 ? 0 : axes.get( i - 1 );
                                int end = axes.get( i ) - 1;
                                return target.error().slice().axis(dim).from(start).to(end).detached();
                            });
                }
            )
            .buildFunAlgorithm()
        );
    }


    @Override public Result execute( Function caller, ExecutionCall<?> call )
    {
        if ( caller.isFlat() && caller.numberOfArgs() != call.inputs().length )
            throw new IllegalArgumentException("The number of arguments of the function call does not match the number of inputs!");

        return super.execute( caller, call );
    }

    private void _catFrames(Tensor<?>[] inputs, Tensor<?> concat, int dim )
    {
        boolean inputsAreFramed = Arrays.stream(inputs).anyMatch( t -> t.frame().isPresent() );

        if ( !inputsAreFramed ) return;

        String label =
                Arrays.stream(inputs)
                .map(Tensor::frame)
                .filter(Optional::isPresent)
                .map(Optional::get)
                .map(NDFrame::getLabel)
                .collect(Collectors.joining("+"));

        if ( !label.isEmpty() ) concat.mut().label(label);

        List<Map<Object, List<Object>>> labels =
                                            Arrays.stream(inputs)
                                                    .map(Tensor::frame)
                                                    .filter(Optional::isPresent)
                                                    .map(Optional::get)
                                                    .map(NDFrame::getState)
                                                    .collect(Collectors.toList());

        List<List<Object>> allKeys = labels.stream().map( l -> new ArrayList<>(l.keySet()) ).collect(Collectors.toList());

        Map<Object, List<Object>> concatFrame = new LinkedHashMap<>();
        for ( int ci = 0; ci < concat.rank(); ci++ ) {

            int finalCi = ci;
            List<Object> distinctKeys = allKeys.stream().map(ks->ks.get(finalCi) ).distinct().collect(Collectors.toList());
            Object key;
            {
                boolean allString = distinctKeys.stream().allMatch(k -> k instanceof String);
                if (allString) // We join using the "+" operator:
                    key = distinctKeys.stream().map(k -> (String) k).collect(Collectors.joining("+"));
                else // We simply take the first one:
                    key = distinctKeys.get(0);
            }

            List<Object> values = new ArrayList<>();
            if ( ci == dim ) {
                /*
                    We need to join the value lists of all the frames
                    and then set the state of the concatenated tensor frame.
                 */
                for ( int i = 0; i < labels.size(); i++ ) {
                    Map<Object, List<Object>> current = labels.get(i);
                    List<Object> currentKeys = allKeys.get(i);
                    List<Object> currentValues = current.get(currentKeys.get(ci));
                    values.addAll(currentValues);
                }
            } else {
                /*
                    This is not as simple as the above case!
                    We have conflicting values for the same key, so we do the following:
                    1. If the values are all equal we just take the first one.
                    2. If the values are not equal but all of type string, we join them with a "+".
                    3. If the values are not equal and not all of type string, we just take the first one.
                 */
                for ( int j = 0; j < concat.shape(ci); j++ ) {
                    List<Object> valuesForThisIndex = new ArrayList<>();
                    for ( int i = 0; i < labels.size(); i++ ) {
                        Map<Object, List<Object>> current = labels.get(i);
                        List<Object> currentKeys = allKeys.get(i);
                        List<Object> currentValues = current.get(currentKeys.get(ci));
                        if ( j < currentValues.size() )
                            valuesForThisIndex.add(currentValues.get(j));
                    }
                    boolean allEqual = valuesForThisIndex.stream().distinct().count() == 1;
                    if ( allEqual )
                        values.add(valuesForThisIndex.get(0));
                    else if ( !valuesForThisIndex.isEmpty() ) {
                        boolean allString = valuesForThisIndex.stream().allMatch( v -> v instanceof String );
                        if ( allString )
                            values.add(valuesForThisIndex.stream().map( v -> (String) v ).collect(Collectors.joining("+")));
                        else
                            values.add(valuesForThisIndex.get(0));
                    }
                }
            }
            concatFrame.put(key, values);
        }
        if ( !concatFrame.isEmpty() ) concat.mut().labelAxes(concatFrame);
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
