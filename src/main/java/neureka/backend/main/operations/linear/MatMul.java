package neureka.backend.main.operations.linear;

import neureka.Neureka;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.backend.main.algorithms.MatMulAlgorithm;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.math.parsing.FunctionParser;

public class MatMul extends AbstractOperation
{
    public MatMul()
    {
        super(
            new OperationBuilder()
                .identifier(       "matMul"    )
                .operator(         "@"         )
                .arity(            2           )
                .isOperator(       true        )
                .isIndexer(        false       )
                .isDifferentiable( true        )
                .isInline(         false       )
        );

        setAlgorithm(
            new MatMulAlgorithm().buildFunAlgorithm()
        );
    }

    @Override
    public Result execute( final Function caller, final ExecutionCall<?> call )
    {
        if ( !caller.isFlat() ) {
            Function reducedCaller = reducePairwise(caller);
            ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flatten( reducedCaller, call.withArgs(Arg.DerivIdx.of(-1)) );
            Function flat = new FunctionParser(Neureka.get().backend()).parse( flatCall.getOperation(), flatCall.arity(), true );
            return super.execute( flat, flatCall );
        }
        return super.execute( reducePairwise(caller), call );
    }

    private Function reducePairwise( final Function fun ) {
        Function reduced = fun;
        if ( reduced.getSubFunctions().size() > 2 ) {
            /*
                So currently we have something like this: a@b@c@d...
                However, this is how it is really executed:  ((((a@b)@c)@d)..)
                ...so let's create a function that is nested like the above:
            */
            Function nested = reduced.getSubFunctions().get(0);
            for ( int i = 1; i < reduced.getSubFunctions().size(); i++ )
                nested = Function.of( nested + " @ " + reduced.getSubFunctions().get(i), true );

            reduced = nested;
        }
        return reduced;
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        return src[ 0 ].call( inputs, j );
    }
}
