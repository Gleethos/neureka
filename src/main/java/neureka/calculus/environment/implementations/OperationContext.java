package neureka.calculus.environment.implementations;

import neureka.calculus.environment.OperationType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class OperationContext implements Cloneable
{
    private static final OperationContext _INSTANCE;
    static {
        _INSTANCE = new OperationContext();
    }

    public static OperationContext instance(){
        return _INSTANCE;
    }

    private final Map<String, OperationType> _LOOKUP;
    private final ArrayList<OperationType> _REGISTER;
    private int _ID;

    private OperationContext(){
        _LOOKUP = new HashMap<>();
        _REGISTER = new ArrayList<>();
        _ID = 0;
    }


    public Map<String, OperationType> getLookup(){
        return _LOOKUP;
    }

    public ArrayList<OperationType> getRegister(){
        return _REGISTER;
    }

    public int getID(){
        return _ID;
    }

    public void incrementID(){
        _ID++;
    }

    public ArrayList<OperationType> instances(){
        return getRegister();
    }

    public OperationType instance(int index){
        return getRegister().get(index);
    }

    public OperationType instance(String identifier){
        return getLookup().getOrDefault(identifier, null);
    }

    @Override
    public OperationContext clone()
    {
        OperationContext clone = new OperationContext();
        clone._ID = _ID;
        clone._LOOKUP.putAll(_LOOKUP);
        clone._REGISTER.addAll(_REGISTER);
        return clone;
    }

}
