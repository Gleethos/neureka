package neureka.calculus.environment;

import neureka.calculus.factory.OperationType;

public class Types
{
    private static Types _types = new Types();

    private Types(){ }

    public static Types instance(){
        Types t = _types;
        _types = null;
        return t;
    }

    public int COUNT(){
        return OperationType.COUNT();
    }

    public String REGISTER(int i){
        return OperationType.REGISTER(i).identifier();
    }

    public int LOOKUP(String exp){
        OperationType type = OperationType.LOOKUP(exp);
        return (type!=null)?type.id():-1;
    }

    public boolean isFunction(String f){
        return isFunction(LOOKUP(f));
    }
    public boolean isFunction(int id){
        OperationType type = OperationType.REGISTER(id);
        return (type!=null)?type.isFunction():false;
    }





}
