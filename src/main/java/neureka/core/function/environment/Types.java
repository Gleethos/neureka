package neureka.core.function.environment;

import java.util.HashMap;

public class Types {

    public String[] REGISTER;
    public HashMap<String, Integer> LOOKUP;

    public Types(){
        REGISTER = new String[]{
                "relu", "sig", "tanh", "quad", "lig", "lin", "gaus", "abs", "sin", "cos",
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

}
