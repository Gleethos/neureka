package neureka;

import java.util.HashMap;
import java.util.Map;

public class NRealm {

    private Map<String, Object> Environment;

    NRealm(){
        Environment = new HashMap<String, Object>();
    }

    public Object get(String name){
        return Environment.get(name);
    }
    public void put(String name, Object thing){
        Environment.put(name, thing);
    }
    public void remove(String name){
        Environment.remove(name);
    }
    public String[] getThingNames(){
        return (String[])Environment.keySet().toArray();
    }



}
