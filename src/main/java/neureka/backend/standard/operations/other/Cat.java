package neureka.backend.standard.operations.other;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.DeviceAlgorithm;
import neureka.backend.api.algorithms.fun.AutoDiffMode;
import neureka.backend.api.algorithms.fun.Result;
import neureka.backend.api.algorithms.fun.SuitabilityPredicate;
import neureka.backend.api.operations.AbstractOperation;
import neureka.backend.api.operations.OperationBuilder;
import neureka.backend.standard.algorithms.FunAlgorithm;
import neureka.calculus.Function;
import neureka.calculus.args.Arg;

import java.util.ArrayList;
import java.util.List;

public class Cat extends AbstractOperation {

    public Cat()
    {
        super(
                new OperationBuilder()
                        .setIdentifier(       "concat"    )
                        .setOperator(         "concat"    )
                        .setArity(            2           )
                        .setIsOperator(       false       )
                        .setIsIndexer(        false       )
                        .setIsDifferentiable( true        )
                        .setIsInline(         false       )
        );
        setAlgorithm(
            FunAlgorithm.class,
            DeviceAlgorithm
            .withName("concat")
            .setIsSuitableFor( call -> {
                Integer dim = call.getValOf(Arg.Dim.class);
                Tsr<?> a = call.input(0);
                Tsr<?> b = call.input(1);
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
                    Integer dim = call.getValOf(Arg.Dim.class);
                    Tsr<?> a = call.input(0);
                    Tsr<?> b = call.input(1);
                    int aAxis = a.shape(dim);
                    int newAxis = aAxis + b.shape(dim);
                    List<Integer> newShape = new ArrayList<>();
                    for ( int i = 0; i < a.rank(); i++ )
                        newShape.add( i == dim ? newAxis : a.shape(i) );

                    Tsr<?> c = Tsr.of( a.getValueClass(), newShape, 0 );
                    Tsr<?> ca = c.slice().axis(dim).from(0).to(aAxis-1).get();
                    Tsr<?> cb = c.slice().axis(dim).from(aAxis).to(newAxis-1).get();
                    List<Integer> caShape = ca.shape();
                    List<Integer> cbShape = cb.shape();
                    Neureka.get().backend().getFunction().idy().execute(ca, a);
                    Neureka.get().backend().getFunction().idy().execute(cb, b);
                    c.getUnsafe().setIsIntermediate(true);
                    return
                        Result.of(c)
                            .withADAction((t, e)->{
                                if ( t.getPayloadShape().equals(caShape) ) {
                                    return e.slice().axis(dim).from(0).to(aAxis-1).get();
                                } else if ( t.getPayloadShape().equals(cbShape) ) {
                                    return e.slice().axis(dim).from(aAxis).to(newAxis-1).get();
                                } else
                                    throw new IllegalArgumentException("Error shape not suitable for back-prop!");
                            });
                }
            )
            .setCallPreparation( call -> call )
            .buildFunAlgorithm()
        );
    }

    @Override
    public String stringify( String[] children ) {
        String expression = String.join( ", ", children );
        if ( expression.charAt( 0 ) == '(' && expression.charAt( expression.length() - 1 ) == ')' ) {
            return "concat" + expression;
        }
        return "concat" + "(" + expression + ")";
    }

    @Override
    public String asDerivative( Function[] children, int derivationIndex) {
        throw new IllegalStateException("Operation does not support dynamic derivation!");
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) { return src[ 0 ].call( inputs, j ); }
}
