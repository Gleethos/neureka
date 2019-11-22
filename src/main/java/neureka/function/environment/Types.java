package neureka.function.environment;

import java.util.HashMap;
import java.util.Map;

public class Types
{
    public String[] REGISTER;
    public Map<String, Integer> LOOKUP;

    public Types()
    {
        REGISTER = new String[]{
                "relu", "sig", "tanh", "quad", "lig", "idy", "gaus", "abs", "sin", "cos",
                "sum", "prod",
                "^", "/", "*", "%", "-", "+", "x", ""+((char)171), ""+((char)187), ",",
                // (char)171 -> <<    // (char)187 -> >>
                "<", ">",
            };
        LOOKUP = new HashMap<>();
        for(int i=0; i<REGISTER.length; i++){
            LOOKUP.put(REGISTER[i], i);
            if(REGISTER[i] == (""+((char)171))){
                LOOKUP.put("<<", i);
            } else if(REGISTER[i] == (""+((char)187))){
                LOOKUP.put(">>", i);
            }
        }
    }

    public boolean isOperation(String f){
        return isOperation(LOOKUP.get(f));
    }
    public boolean isOperation(int id){
        return (id>=LOOKUP.get("^"));
    }

    public boolean isFunction(String f){
        return isFunction(LOOKUP.get(f));
    }
    public boolean isFunction(int id){
        return (id<=LOOKUP.get("cos"));
    }

    public boolean isIndexer(String f){
        return isIndexer(LOOKUP.get(f));
    }
    public boolean isIndexer(int id){
        return (id>=LOOKUP.get("sum"))&&(id<=LOOKUP.get("prod"));
    }

    public boolean isConvection(String f){
        return isConvection(LOOKUP.get(f));
    }
    public boolean isConvection(int id){
        return (REGISTER[id] == "x" || REGISTER[id] == "«" || REGISTER[id] == "»");
    }

    public boolean isCommutative(String f){
        switch(f){
            case "^": return false;
            case "/": return  false;
            case "*": return true;
            case "%": return false;
            case "-": return false;
            case "+": return true;
            case "x": return false;
            case (""+((char)171)): return false;
            case (""+((char)187)): return false;
            case ",":return false;
            case "<": return false;
            case ">": return false;
        }
        return false;
    }
    public boolean isCommutative(int id){
        return isCommutative(REGISTER[id]);
    }

}
