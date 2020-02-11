package neureka.framing;

import neureka.Tsr;

import java.util.*;
import java.util.function.Function;

public class Index {

    /*
        index is a 2d array.
     */

    private Map<Object, Map<Object, Integer>> _mapping;

    public Index(List<List> labels){
        _mapping = new LinkedHashMap<>(labels.size());
        for(int i=0; i<labels.size(); i++) _mapping.put(i, new LinkedHashMap<>());
        for(int i=0; i<labels.size(); i++){
            if(labels.get(i)!=null){
                for(int ii=0; ii<labels.get(i).size(); ii++){
                    if(labels.get(i).get(ii)!=null) set(labels.get(i).get(ii), i, ii);
                }
            }
        }
    }

    public Index(int size) {
        _mapping = new LinkedHashMap<>(size);
        for(int i=0; i<size; i++) _mapping.put(i, new LinkedHashMap<>());
    }

    public Index(Map<Object, List<Object>> labels, Tsr host){
        _mapping = new LinkedHashMap<>(labels.size()*3);
        int[] index = {0};
        labels.forEach((k, v)->{
            if(v!=null){
                Map<Object, Integer> idxmap = new LinkedHashMap<>(v.size()*3);
                for(int i=0; i<v.size(); i++) idxmap.put(v.get(i), i);
                _mapping.put(k, idxmap);
            } else {
                int size = host.shape()[index[0]];
                Map<Object, Integer> idxmap = new LinkedHashMap<>(size);
                for(int i=0; i<size; i++) idxmap.put(i, i);
                _mapping.put(k, idxmap);
            }
            index[0]++;
        });
        //Object[] asArray = axisLabels.toArray();
        //_mapping = new HashMap<>(axisLabels.size());
        //for(int i=0; i<axisLabels.size(); i++) _mapping.put(asArray[i], new HashMap<>());//[i] = (new HashMap<>());
    }

    public int[] get(List<Object> keys){
        return get(keys.toArray(new Object[keys.size()]));
    }

    public int[] get(Object[] keys){
        int[] idx = new int[keys.length];//_mapping.length];
        for(int i=0; i<idx.length; i++) idx[i] = _mapping.get(i).get(keys[i]);
        return idx;
    }

    public int get(Object key, Object axis){
        return _mapping.get(axis).get(key);
    }

    public void set(Object indexKey, Object axis, int index){
        _mapping.get(axis).put(indexKey, index);
    }

    public List<Object> keysOf(Object axis, int index){
        List<Object> keys = new ArrayList<>();
        _mapping.get(axis).forEach((k, v)->{if(v==index) keys.add(k);});
        return keys;
    }

    private String _fixed(String str, int size){
        if(str.length()<size){
            int first = size/2;
            int second = size - first;
            first -= str.length()/2;
            second -= str.length()-str.length()/2;
            for (int i=0; i<first; i++) str = " "+str;
            for (int i=0; i<second; i++) str += " ";
        }
        return str;
    }


    @Override
    public String toString(){
        final int WIDTH = 16;
        final String WALL = " | ";
        final String HEADLINE = "=";
        final String ROWLINE = "-";
        final String CROSS = "+";

        int indexShift = (WALL.length()/2);
        int crossMod = WIDTH+WALL.length();
        Function<Integer, Boolean> isCross = (i)->(i-indexShift)%crossMod==0;

        StringBuilder builder = new StringBuilder();

        //builder.append("Tensor Index: axis/indexes\n");

        builder.append(WALL);

        int[] axisLabelSizes = new int[_mapping.size()];
        int[] axisCounter = {0};
        _mapping.forEach((k, v)->{
            String axisHeader = k.toString();
            axisHeader = _fixed(axisHeader, WIDTH);
            axisLabelSizes[axisCounter[0]] = axisHeader.length();
            builder.append(axisHeader);
            builder.append(WALL);
            axisCounter[0]++;
        });
        int lineLength = builder.length();
        builder.append("\n");
        for(int i=0; i<lineLength; i++) builder.append((isCross.apply(i))?CROSS:HEADLINE);
        builder.append("\n");
        boolean[] hasMoreIndexes = {true};
        int[] depth = {0};
        while (hasMoreIndexes[0]){
            axisCounter[0] = 0;
            Object[] keyOfDepth = {null};
            builder.append(WALL);
            _mapping.forEach((k, v)->{
                keyOfDepth[0] = null;
                //if(v!=null){
                    v.forEach((ik, iv)->{
                        if(iv.intValue()==depth[0]){
                            keyOfDepth[0] = ik;
                        }
                    });
                //}
                if(keyOfDepth[0]!=null){
                    builder.append(_fixed((keyOfDepth[0]).toString(), WIDTH));
                } else {
                    //for(int i=0; i<axisLabelSizes[axisCounter[0]]; i++) builder.append(" ");
                    builder.append(_fixed("---", WIDTH));
                }
                builder.append(WALL);
                axisCounter[0] ++;
            });
            depth[0]++;
            builder.append("\n");
            for(int i=0; i<lineLength; i++) builder.append((isCross.apply(i))?CROSS:ROWLINE);
            builder.append("\n");
            if(keyOfDepth[0]==null) hasMoreIndexes[0] = false;
        }

        StringBuilder result = new StringBuilder().append("\nTensor Index: axis/indexes");
        result.append("\n");
        for(int i=0; i<lineLength; i++) result.append(HEADLINE);
        result.append("\n");

        result.append(builder);


        return result.toString();

    }




}
