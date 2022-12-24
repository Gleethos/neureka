package neureka.backend.main.operations.indexer;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.Operation;
import neureka.backend.api.Result;
import neureka.backend.api.template.algorithms.AbstractDeviceAlgorithm;
import neureka.backend.api.template.operations.AbstractOperation;
import neureka.backend.api.template.operations.OperationBuilder;
import neureka.math.Function;
import neureka.math.args.Arg;
import neureka.math.parsing.FunctionParser;

/**
 *  This type of operation belongs to the same species as the
 *  {@link Product} operation.
 *  It executes incoming calls so that the calling function
 *  will be executed with all input indices passed to it.
 *  The resulting array of tensors will then be summed
 *  to produce the result of this operation, hence the name {@link Summation}.
 */
public final class Summation extends AbstractOperation
{
    public Summation()
    {
        super (
            new OperationBuilder()
            .identifier(       "sumJs" )
            .operator(         "sumJs" )
            .arity(            1       )
            .isOperator(       false   )
            .isIndexer(        true    )
            .isDifferentiable( true    )
            .isInline(         false   )
        );
        /*
            The summation operation does not have algorithms because it is
            a special derivative case of the "addition" operation.
         */
    }

    @Override
    public Result execute( final Function caller, final ExecutionCall<?> call )
    {
        Tsr<?>[] inputs = new Tsr[ call.arity() ];
        for ( int i = 0; i < inputs.length; i++ ) {
            ExecutionCall<?> flatCall = AbstractDeviceAlgorithm.flattenForIndexer( caller, call.withArgs(Arg.VarIdx.of(i)) );
            inputs[ i ] = flatCall.input( 0 );
        }
        Operation plusOp = Neureka.get().backend().getOperation("+");
        Function plus = new FunctionParser(Neureka.get().backend())
                                .parse( plusOp, inputs.length, caller.isDoingAD() );

        return plusOp.execute( plus, call.withInputs(inputs).withOperation(plusOp).withArgs(Arg.DerivIdx.of(-1)) );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, Function[] src ) {
        if ( j < 0 ) return calculate( inputs, d, src );
        if ( d < 0 ) return _calculate( inputs, src );
        else return src[ 0 ].derive( inputs, d, j );
    }

    
    public static double calculate( double[] inputs, int d, Function[] src ) {
        if ( d < 0 )
            return _calculate( inputs, src );
        else {
            double sum = 0;
            boolean nothingDone = true;
            for ( int i = 0; i < inputs.length; i++ ) {
                double r = src[ 0 ].derive( inputs, d, i );
                sum += r;
                nothingDone = false;
            }
            if ( nothingDone ) return src[ 0 ].call( inputs );
            return sum;
        }

    }

    private static double _calculate( double[] inputs, Function[] src ) {
        double sum = 0;
        boolean nothingDone = true;
        for ( int i = 0; i < inputs.length; i++ ) {
            sum += src[ 0 ].call( inputs, i );
            nothingDone = false;
        }
        if ( nothingDone ) return src[ 0 ].call( inputs );
        return sum;
    }


}
