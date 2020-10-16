package neureka.calculus.backend.operations;

import neureka.calculus.Function;

import java.util.ArrayList;
import java.util.List;

public class OperationTypeFactory
{
    String _function = null;
    String _operator = null;
    Integer _arity = null;
    Boolean _isOperator = null;
    Boolean _isIndexer = null;
    Boolean _isDifferentiable = null;
    Boolean _isInline = null;

    public OperationTypeFactory(){

    }

    public AbstractOperationType create(){
        List<String> missing = new ArrayList<>();
        if( _function == null ) missing.add("function");
        if( _operator == null ) missing.add("operator");
        if( _arity == null ) missing.add("arity");
        if( _isOperator == null ) missing.add("isOperator");
        if( _isIndexer == null ) missing.add("isIndexer");
        if( _isDifferentiable == null ) missing.add("isDifferentiable");
        if( _isInline == null ) missing.add("isInline");
        AbstractOperationType result = null;
        if( !missing.isEmpty() ) {
            throw new IllegalStateException("Factory not satisfied! The following properties are missing: '"+ String.join(", ", missing) +"'");
        } else {
            result = new AbstractOperationType (
                    _function,
                    _operator,
                    _arity,
                    _isOperator,
                    _isIndexer,
                    _isDifferentiable,
                    _isInline
            ) {
                @Override
                public double calculate( double[] inputs, int j, int d, List<Function> src ) {
                    return src.get( 0 ).call( inputs, j );
                }
            };
        }
        return result;
    }

    public OperationTypeFactory withFunction( String function ) {
        _function = function;
        return this;
    }


    public OperationTypeFactory withOperator( String operator ) {
        _operator = operator;
        return this;
    }


    public OperationTypeFactory withArity( int arity ) {
        _arity = arity;
        return this;
    }

    public OperationTypeFactory setIsOperator( boolean isOperator ) {
        _isOperator = isOperator;
        return this;
    }

    public OperationTypeFactory setIsIndexer( boolean isIndexer ) {
        _isIndexer = isIndexer;
        return this;
    }

    public OperationTypeFactory setIsDifferentiable( boolean isDifferentiable ) {
        _isDifferentiable = isDifferentiable;
        return this;
    }

    public OperationTypeFactory setIsInline( boolean isInline ) {
        _isInline = isInline;
        return this;
    }




}



