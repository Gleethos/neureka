package neureka.framing;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Index {

    /*
        index is a 2d array.

     */

    private Map<Object, Integer>[] _mapping;

    public Index(int size){
        _mapping = new Map[size];//new ArrayList<>();
        for(int i=0; i<size; i++) _mapping[i] = (new HashMap<>());
    }

    public int[] get(List<Object> keys){
        return get(keys.toArray(new Object[keys.size()]));
    }

    public int[] get(Object[] keys){
        int[] idx = new int[_mapping.length];
        for(int i=0; i<idx.length; i++) idx[i] = _mapping[i].get(keys[i]);
        return idx;
    }

    public int get(Object key, int axis){
        return _mapping[axis].get(key);
    }

    public void set(Object indexKey, int axis, int index){
        _mapping[axis].put(indexKey, index);
    }

    public List<Object> keysOf(int axis, int index){
        List<Object> keys = new ArrayList<>();
        _mapping[axis].forEach((k, v)->{if(v==index) keys.add(k);});
        return keys;
    }





}
