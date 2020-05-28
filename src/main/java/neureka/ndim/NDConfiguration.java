package neureka.ndim;

import neureka.ndim.config.D1Configuration;
import neureka.ndim.config.DefaultNDConfiguration;

public interface NDConfiguration
{
    int rank();

    int[] shape();

    int shape(int i);

    int[] idxmap();

    int idxmap(int i);

    int[] translation();

    int translation(int i);

    int[] spread();

    int spread(int i);

    int[] offset();

    int offset(int i);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int i_of_i(int i);

    int[] idx_of_i(int i);

    int i_of_idx(int[] idx);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    static NDConfiguration construct(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ){
        if(shape.length==1) return D1Configuration.construct(shape, translation, idxmap, spread, offset);
        return DefaultNDConfiguration.construct(shape, translation, idxmap, spread, offset);
    }


}
