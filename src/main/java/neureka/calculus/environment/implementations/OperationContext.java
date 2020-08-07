package neureka.calculus.environment.implementations;

import neureka.calculus.environment.OperationType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class OperationContext
{

    private final Map<String, OperationType> _LOOKUP = new HashMap<>();

    private final ArrayList<OperationType> _REGISTER = new ArrayList<>();

    private int _ID = 0;

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

}
