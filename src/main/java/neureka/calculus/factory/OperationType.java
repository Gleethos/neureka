package neureka.calculus.factory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class OperationType {

    private static Map<String, OperationType> _LOOKUP = new HashMap<>();

    private static ArrayList<OperationType> _REGISTER = new ArrayList<>();

    private static int _ID = 0;

    public static OperationType LOOKUP(String identifier){
        return _LOOKUP.getOrDefault(identifier, null);
    }

    public static OperationType REGISTER(int index){
        return _REGISTER.get(index);
    }

    public static int COUNT(){
        return _ID;
    }

    private int _id;
    private String  _identifier;
    private boolean _isFunction;
    private boolean _isOperation;
    private boolean _isIndexer;
    private boolean _isConvection;
    private boolean _isCommutative;
    private boolean _isAssociative;

    public OperationType(
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative
    ) {
        _construct(
                identifier,
                isFunction,
                isOperation,
                isIndexer,
                isConvection,
                isCommutative,
                isAssociative
        );
    }

    public OperationType(
            String  identifier,
            boolean isFunction,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative
    ) {
        _construct(
                identifier,
                isFunction,
                !isFunction,
                isIndexer,
                isConvection,
                isCommutative,
                isAssociative
        );
    }

    private void _construct(
            String  identifier,
            boolean isFunction,
            boolean isOperation,
            boolean isIndexer,
            boolean isConvection,
            boolean isCommutative,
            boolean isAssociative
    ) {
        _id = _ID;
        _ID++;
        _identifier = identifier;
        _isFunction = isFunction;
        _isOperation = isOperation;
        _isIndexer = isIndexer;
        _isConvection = isConvection;
        _isCommutative = isCommutative;
        _isAssociative = isAssociative;

        _REGISTER.add(this);
        _LOOKUP.put(identifier, this);
        if(identifier.equals((((char)171))+"x")) _LOOKUP.put("<<x", this);
        else if(identifier.equals("x"+((char)187))) _LOOKUP.put("x>>", this);
    }

    public int id(){
        return _id;
    }

    public String identifier(){
        return _identifier;
    }

    public boolean isOperation(){
        return _isOperation;
    }

    public boolean isFunction(){
        return _isFunction;
    }

    public boolean isIndexer(){
        return _isIndexer;
    }

    public boolean isConvection(){
        return _isConvection;
    }

    public boolean isCommutative(){
        return  _isCommutative;
    }

}
