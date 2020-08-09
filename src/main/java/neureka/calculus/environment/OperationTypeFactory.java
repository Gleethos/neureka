package neureka.calculus.environment;

import java.awt.datatransfer.StringSelection;
import java.util.ArrayList;
import java.util.List;

public class OperationTypeFactory
{
    String  _name = null;
    String  _identifier = null;
    Integer _arity = null;
    Boolean _isOperation = null;
    Boolean _isIndexer = null;
    Boolean _isConvection = null;
    Boolean _isCommutative = null;
    Boolean _isAssociative = null;

    public OperationTypeFactory(){

    }

    public OperationType create(){
        List<String> missing = new ArrayList<>();
        if( _name == null ) missing.add("name");
        if( _name == null ) missing.add("identifier");
        if( _name == null ) missing.add("arity");
        if( _name == null ) missing.add("isOperation");
        if( _name == null ) missing.add("isIndexer");
        if( _name == null ) missing.add("isConvection");
        if( _name == null ) missing.add("isCommutative");
        if( _name == null ) missing.add("isAssociative");
        OperationType result = null;
        if( !missing.isEmpty() ) {
            throw new IllegalStateException("Factory not satisfied! The following properties are missing: '"+ String.join(", ", missing) +"'");
        } else {
            result = new OperationType(
                    _name,
                    _identifier,
                    _arity,
                    _isOperation,
                    _isIndexer,
                    _isConvection,
                    _isCommutative,
                    _isAssociative
            );
        }
        return result;
    }

}



