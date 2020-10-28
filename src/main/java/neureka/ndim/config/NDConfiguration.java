package neureka.ndim.config;

import neureka.Neureka;
import neureka.ndim.AbstractNDArray;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

public interface NDConfiguration
{
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    int[] asInlineArray();

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    long keyCode();

    boolean equals(NDConfiguration ndc);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    NDConfiguration newReshaped(int[] newForm);

    public class Utility {

        @Contract(pure = true)
        public static int[] rearrange(int[] tln, int[] shp, @NotNull int[] newForm) {
            int[] shpTln = newTlnOf(shp);
            int[] newTln = new int[newForm.length];
            for (int i = 0; i < newForm.length; i++) {
                if (newForm[ i ] < 0) newTln[ i ] = shpTln[ i ];
                else if (newForm[ i ] >= 0) newTln[ i ] = tln[newForm[ i ]];
            }
            return newTln;
        }

        @Contract(pure = true)
        public static int[] rearrange(@NotNull int[] array, @NotNull int[] ptr) {
            int[] newShp = new int[ptr.length];
            for (int i = 0; i < ptr.length; i++) {
                if (ptr[ i ] < 0) newShp[ i ] = Math.abs(ptr[ i ]);
                else if (ptr[ i ] >= 0) newShp[ i ] = array[ptr[ i ]];
            }
            return newShp;
        }

        @Contract(pure = true)
        public static int[] newTlnOf(int[] shape) {
            int[] tln = new int[ shape.length ];
            int prod = 1;
            if (Neureka.instance().settings().indexing().isUsingLegacyIndexing()) {
                for (int i = 0; i < tln.length; i++) {
                    tln[ i ] = prod;
                    prod *= shape[ i ];
                }
            } else {
                for (int i = tln.length-1; i >= 0; i--) {
                    tln[ i ] = prod;
                    prod *= shape[ i ];
                }
            }
            return tln;
        }

        @Contract(pure = true)
        public static void increment(@NotNull int[] shpIdx, @NotNull int[] shape) {
            int i;
            if (Neureka.instance().settings().indexing().isUsingLegacyIndexing()) i = 0;
            else i = shape.length-1;
            while (i >= 0 && i < shape.length) i = _incrementAt(i, shpIdx, shape);
        }

        @Contract(pure = true)
        private static int _incrementAt(int i, @NotNull int[] shpIdx, @NotNull int[] shape) {
            if ( Neureka.instance().settings().indexing().isUsingLegacyIndexing() ) {
                if (shpIdx[ i ] < (shape[ i ])) {
                    shpIdx[ i ]++;
                    if (shpIdx[ i ] == (shape[ i ])) {
                        shpIdx[ i ] = 0;
                        i++;
                    } else {
                        i = -1;
                    }
                } else {
                    i++;
                }
                return i;
            } else {
                if (shpIdx[ i ] < (shape[ i ])) {
                    shpIdx[ i ]++;
                    if (shpIdx[ i ] == (shape[ i ])) {
                        shpIdx[ i ] = 0;
                        i--;
                    } else {
                        i = -1;
                    }
                } else {
                    i--;
                }
                return i;
            }
        }

        @Contract(pure = true)
        public static int szeOfShp(int[] shape) {
            int size = 1;
            for (int i : shape) size *= i;
            return size;
        }


    }

}
