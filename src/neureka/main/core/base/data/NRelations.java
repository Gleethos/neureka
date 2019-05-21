package neureka.main.core.base.data;

import java.util.Map;
import java.util.function.BiConsumer;

public class NRelations {

    Map<NTensor, NTensor> map;

    NRelations(){}

    public void forEach(BiConsumer<NTensor, NTensor> action){
        map.forEach(action);
    }
    public void add(NTensor key, NTensor targ){
        map.put(key, targ);
    }

}
