package neureka.backend.api.operations;

import lombok.Setter;
import lombok.experimental.Accessors;

import java.util.ArrayList;
import java.util.List;

@Accessors( prefix = {"_"}, chain = true )
public class OperationTypeFactory
{
    @Setter String _function = null;
    @Setter String _operator = null;
    @Setter Integer _arity = null;
    @Setter Boolean _isOperator = null;
    @Setter Boolean _isIndexer = null;
    @Setter Boolean _isDifferentiable = null;
    @Setter Boolean _isInline = null;


    public GenericOperation build()
    {
        List<String> missing = new ArrayList<>();
        if ( _function == null ) missing.add( "function" );
        if ( _operator == null ) missing.add( "operator" );
        if ( _arity == null ) missing.add( "arity" );
        if ( _isOperator == null ) missing.add( "isOperator" );
        if ( _isIndexer == null ) missing.add( "isIndexer" );
        if ( _isDifferentiable == null ) missing.add( "isDifferentiable" );
        if ( _isInline == null ) missing.add( "isInline" );

        if ( !missing.isEmpty() )
            throw new IllegalStateException("Factory not satisfied! The following properties are missing: '"+ String.join(", ", missing) +"'");
        else
            return new GenericOperation (
                    _function,
                    _operator,
                    _arity,
                    _isOperator,
                    _isIndexer,
                    _isDifferentiable,
                    _isInline
            );
    }

}



