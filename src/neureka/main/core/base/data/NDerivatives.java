package neureka.main.core.base.data;

import java.util.HashMap;
import java.util.Set;
import java.util.function.BiConsumer;

public class NDerivatives {

    private HashMap<NTensor, NTensor> map = new HashMap<NTensor, NTensor>();
    public NDerivatives(){ }
    public void put(NTensor key, NTensor value){
        map.put(key, value);
    }
    public NTensor get(NTensor key){
        return map.get(key);
    }
    public boolean has(NTensor key){
        return map.containsKey(key);
    }
    public Set<NTensor> sources(){
        return map.keySet();
    }
    public void forEach(BiConsumer<NTensor, NTensor> action){
        map.forEach(action);
    }

}
