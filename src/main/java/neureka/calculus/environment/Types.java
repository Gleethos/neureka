package neureka.calculus.environment;

import neureka.autograd.ADAgent;
import neureka.calculus.factory.OperationType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Types
{
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

    public List<OperationType> getAll(){
        return _types;
    }

    private ArrayList<OperationType> _types;

    /**
     *  // *,    /,    ^,    +,    -,
     *  // x,    d,    p,    a,    s,
     *  // <<, <<d,  <<p,  <<a,  <<s,
     */
    public Types()
    {
        OperationType[] types = new OperationType[]{
                new OperationType("relu", true, false, false, true, true),
                new OperationType("sig" , true, false, false, true, true),
                new OperationType("tanh", true, false, false, true, true),
                new OperationType("quad", true, false, false, true, true),
                new OperationType("lig" , true, false, false, true, true),
                new OperationType("idy" , true, false, false, true, true),
                new OperationType("gaus", true, false, false, true, true),
                new OperationType("abs" , true, false, false, true, true),
                new OperationType("sin" , true, false, false, true, true),
                new OperationType("cos" , true, false, false, true, true),

                // Indexer:
                new OperationType("sum" , false, false, true, false, true, true),
                new OperationType("prod", false, false,  true, false, true, true),

                // Operations (auto broadcast):
                new OperationType("^", false, false, false, false, false),
                new OperationType(((char)171)+"^", false, false, false, false, false),
                new OperationType("^"+((char)187), false, false, false, false, false),

                new OperationType("/", false, false, false, false, false),
                new OperationType(((char)171)+"/", false, false, false, false, false),
                new OperationType("/"+((char)187), false, false, false, false, false),

                new OperationType("*", false, false, false, true, false),
                new OperationType(((char)171)+"*", false, false, false, false, false),
                new OperationType("*"+((char)187), false, false, false, false, false),

                new OperationType("%", false, false, false, false, false),
                new OperationType(((char)171)+"%", false, false, false, false, false),
                new OperationType("%"+((char)187), false, false, false, false, false),

                new OperationType("-", false, false, false, false, false),
                new OperationType(((char)171)+"-", false, false, false, false, false),
                new OperationType("-"+((char)187), false, false, false, false, false),

                new OperationType("+", false, false, false, true, false),
                new OperationType(((char)171)+"+", false, false, false, false, false),
                new OperationType("+"+((char)187), false, false, false, false, false),

                // Convolution:
                new OperationType("x", false, false, true, false, false),
                new OperationType(((char)171)+"x", false, false, true, false, false),
                new OperationType("x"+((char)187), false, false, true, false, false),

                new OperationType("d", false, false, true, false, false),
                new OperationType(((char)171)+"d", false, false, true, false, false),
                new OperationType("d"+((char)187), false, false, true, false, false),

                new OperationType("p", false, false, true, false, false),
                new OperationType(((char)171)+"p", false, false, true, false, false),
                new OperationType("p"+((char)187), false, false, true, false, false),

                new OperationType("a", false, false, true, false, false),
                new OperationType(((char)171)+"a", false, false, true, false, false),
                new OperationType("a"+((char)187), false, false, true, false, false),

                new OperationType("s", false, false, true, false, false),
                new OperationType(((char)171)+"s", false, false, true, false, false),
                new OperationType("s"+((char)187), false, false, true, false, false),
                // (char)171 -> <<    // (char)187 -> >>

                // Reshape:
                new OperationType(",", false, false, false, false, false),

                // Injecting:
                new OperationType("<", false, false, false, false, false),
                new OperationType(">", false, false, false, false, false),
        };
        _types = new ArrayList<>();
        for(OperationType type : types) _types.add(type);

    }


    public boolean isOperation(String f){
        return isOperation(LOOKUP(f));
    }
    public boolean isOperation(int id){
        OperationType type = OperationType.REGISTER(id);
        return (type!=null)?type.isOperation():false;
    }

    public boolean isFunction(String f){
        return isFunction(LOOKUP(f));
    }
    public boolean isFunction(int id){
        OperationType type = OperationType.REGISTER(id);
        return (type!=null)?type.isFunction():false;
    }

    public boolean isIndexer(String f){
        return isIndexer(LOOKUP(f));
    }
    public boolean isIndexer(int id){
        OperationType type = OperationType.REGISTER(id);
        return (type!=null)?type.isIndexer():false;
    }

    public boolean isConvection(String f){
        return isConvection(LOOKUP(f));
    }
    public boolean isConvection(int id){
        OperationType type = OperationType.REGISTER(id);
        return (type!=null)?type.isConvection():false;
        //return (REGISTER[id].equals("x") || REGISTER[id].contains("«") || REGISTER[id].contains("»"));
    }

    public boolean isCommutative(String f){
        OperationType type = OperationType.LOOKUP(f);
        return (type!=null)?type.isCommutative():false;
    }
    public boolean isCommutative(int id){
        OperationType type = OperationType.REGISTER(id);
        return (type!=null)?type.isCommutative():false;
    }


}
