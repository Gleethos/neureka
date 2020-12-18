package neureka.backend.api.operations;

import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;
import neureka.calculus.Function;

import java.util.List;

@NoArgsConstructor
@Accessors( prefix = {"_"} )
public class GenericOperation extends AbstractOperationType
{
    public GenericOperation(
            String function,
            String operator,
            Integer arity,
            Boolean isOperator,
            Boolean isIndexer,
            Boolean isDifferentiable,
            Boolean isInline
    ) {
        super( function, operator, arity, isOperator, isIndexer, isDifferentiable, isInline );
    }

    @Override
    public double calculate( double[] inputs, int j, int d, List<Function> src ) {
        return src.get( 0 ).call( inputs, j );
    }
}
