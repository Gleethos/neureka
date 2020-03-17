package neureka.acceleration;

import neureka.Tsr;
import neureka.calculus.environment.OperationType;
import neureka.calculus.environment.Type;
import org.jetbrains.annotations.Contract;

import java.util.Arrays;
import java.util.Collection;

public class CPU extends AbstractDevice {

    @Override
    protected void _enqueue(Tsr[] tsrs, int d, OperationType type) {
        for(Tsr t : tsrs) t.setIsVirtual(false);
        if(type.supportsActivation() && !type.isIndexer()){
            Exec.activate(tsrs, d, type);
            return;
        }
        if(type.isOperation()&&!type.isConvection()) {
            Exec.broadcast(tsrs, d, type);
            return;
        }
        if(type.isConvection()){
            if(type.identifier().contains(((char) 187)+"")) Exec.convolve(new Tsr[]{tsrs[2], tsrs[1], tsrs[0]}, d, type);
            else if(type.identifier().contains(((char) 171)+"")) Exec.convolve(new Tsr[]{tsrs[0], tsrs[1], tsrs[2]}, d, type);
            else
            {
                if (d >= 0) {
                    if (d == 0) tsrs[0] = tsrs[2]; else tsrs[0] = tsrs[1];
                } else {
                    Exec.convolve(tsrs, -1, type);
                }
            }

        } else if (type.isIndexer()) Exec.broadcast(tsrs, d, type);
    }

    @Override
    protected void _enqueue(Tsr t, double value, int d, OperationType type) {
        if(type.supportsScalar()){
            Exec.scalar(new Tsr[]{t, t}, value, d, type);
            return;
        }
        int[] shape = new int[t.rank()];
        Arrays.fill(shape, 1);
        _enqueue(new Tsr[]{t, t, new Tsr(shape, value)}, d, type);
    }

    @Override
    public void dispose() {
    }

    @Override
    public Device get(Tsr tensor) {
        return this;
    }

    @Override
    public Device add(Tsr tensor) {
        return this;
    }

    @Override
    public Device add(Tsr tensor, Tsr parent) {
        return this;
    }

    @Override
    public boolean has(Tsr tensor) {
        return false;
    }

    @Override
    public Device rmv(Tsr tensor) {
        return this;
    }

    @Override
    public Device overwrite64(Tsr tensor, double[] value) {
        return this;
    }

    @Override
    public Device overwrite32(Tsr tensor, float[] value) {
        return this;
    }

    @Override
    public Device swap(Tsr former, Tsr replacement) {
        return this;
    }

    @Override
    public double[] value64Of(Tsr tensor) {
        return tensor.value64();
    }

    @Override
    public float[] value32Of(Tsr tensor) {
        return tensor.value32();
    }

    @Override
    public Collection<Tsr> tensors() {
        return null;
    }

    public static class Exec
    {
        interface Range {
            void execute(int start, int end);
        }

        //---
        public static void activate(Tsr[] tsrs, int d, OperationType type){
            _threaded(tsrs[0].size(),
                    (start, end) ->
                    _template.activate(
                            tsrs[0], start, end,
                            type.getActivation().getCreator().create(tsrs, d)
                    )
            );
        }

        //---

        public static void broadcast(Tsr[] tsrs, int d, OperationType type){
            _threaded(tsrs[0].size(), (start, end) ->
                _template.broadcast(
                        tsrs[0], tsrs[1], tsrs[2], d,
                        start, end,
                        type.getBroadcast().getCreator().create(tsrs,  d)
                )
            );
        }

        //---

        public static void convolve(Tsr[] tsrs, int d, OperationType type){
            _threaded(tsrs[0].size(), (start, end) ->
                _template.convolve(
                        tsrs[0], tsrs[1], tsrs[2], d,
                        start, end,
                        type.getConvolution().getCreator().create(tsrs,  -1)
                )
            );
        }

        //---

        public static void scalar(Tsr[] tsrs, double scalar, int d, OperationType type){
            _threaded(tsrs[0].size(), (start, end) ->
                    _template.activate(
                        tsrs[0], start, end,
                        type.getScalarization().getCreator().create(tsrs, scalar, d)
                )
            );
        }

        //==============================================================================================================

        private static void _threaded(int sze, Range range) {
            boolean doThreading = false;
            if (sze > 128) {
                doThreading = ((sze / Runtime.getRuntime().availableProcessors()) > 32);
            }
            if (!doThreading) {
                range.execute(0, sze);
            } else {
                int threadCount = Runtime.getRuntime().availableProcessors();
                final int chunk = (sze / threadCount);
                Thread[] th = new Thread[threadCount];
                for (int i = 0; i < threadCount; i++) {
                    final int start = i * chunk;
                    final int end = (i == threadCount - 1) ? sze : ((i + 1) * chunk);
                    th[i] = new Thread(() -> range.execute(start, end));
                    th[i].start();
                }
                for (int i = 0; i < threadCount; i++) {
                    try {
                        th[i].join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }


        private static class _template {
            @Contract(pure = true)
            public static void convolve(
                    Tsr t0_drn, Tsr t1_src, Tsr t2_src,
                    int d,
                    int i, int end,
                    Type.Operator operation
            ) {
                int[] t0Shp = t0_drn.shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
                int[] t1Shp = t1_src.shape();
                int[] t2Shp = t2_src.shape();
                int rank = t0Shp.length;
                int[] t0Idx = new int[rank];
                int[] t1Idx = new int[rank];
                int[] t2Idx = new int[rank];
                double[] t0_value = t0_drn.value64();
                //double[] t1_value = t1_src.value64();
                //double[] t2_value = t2_src.value64();
                //int drnSze = t0_drn.size();
                //int i = 0;

                if (d < 0) {
                    while (i < end)//drnSze)
                    {//increment on drain accordingly:
                        int ri = 0;
                        while (ri < rank) {
                            if (t1Shp[ri] == t2Shp[ri]) {
                                t1Idx[ri] = t0Idx[ri];
                                t2Idx[ri] = t0Idx[ri];
                            } else if (t1Shp[ri] > t2Shp[ri]) {
                                t1Idx[ri] = t0Idx[ri];
                                t2Idx[ri] = 0;
                            } else if (t1Shp[ri] < t2Shp[ri]) {
                                t1Idx[ri] = 0;
                                t2Idx[ri] = t0Idx[ri];
                            }
                            ri++;
                        }
                        //----------
                        // multiplication:
                        double value = 0;
                        boolean running = true;
                        boolean incrementing = false;
                        while (running) {
                            ri = (ri == rank) ? 0 : ri;
                            if (!incrementing) {
                                value += operation.execute(t0Idx, t1Idx, t2Idx);
                                incrementing = true;
                                ri = 0;
                            } else {//incrementing:
                                if (t1Idx[ri] < t1Shp[ri] && t2Idx[ri] < t2Shp[ri]) {
                                    t1Idx[ri]++;
                                    t2Idx[ri]++;
                                    if (t1Idx[ri] == t1Shp[ri] || t2Idx[ri] == t2Shp[ri]) {
                                        running = running && !(ri == (rank - 1));
                                        if (t1Shp[ri] == t2Shp[ri]) {
                                            t1Idx[ri] = t0Idx[ri];
                                            t2Idx[ri] = t0Idx[ri];
                                        } else if (t1Shp[ri] > t2Shp[ri]) {
                                            t1Idx[ri] = t0Idx[ri];
                                            t2Idx[ri] = 0;
                                        } else if (t1Shp[ri] < t2Shp[ri]) {
                                            t1Idx[ri] = 0;
                                            t2Idx[ri] = t0Idx[ri];
                                        }
                                        ri++;
                                    } else {
                                        incrementing = false;
                                    }
                                } else {
                                    ri++;
                                }
                            }
                        }//setInto _value in drn:
                        t0_value[t0_drn.i_of_idx(t0Idx)] = value;
                        //increment on drain:
                        Tsr.Utility.Indexing.increment(t0Idx, t0Shp);
                        i++;
                    }
                }
                else//---
                {
                    while (i < end) {//increment on drain accordingly:
                        int ri = 0;
                        while (ri < rank) {
                            if (t2Idx[ri] == t2Shp[ri]) {//setting 0
                                t1Idx[ri] = t0Idx[ri];
                                t2Idx[ri] = 0;//mtch[mi];
                            } else {
                                if (t0Shp[ri] > t1Shp[ri]) {
                                    t1Idx[ri] = (t0Idx[ri] - t2Idx[ri]);
                                } else {
                                    t1Idx[ri] = (t0Idx[ri] + t2Idx[ri]);
                                }
                            }
                            ri++;
                        }
                        //----------
                        // multiplication:
                        double value = 0;
                        boolean running = true;
                        boolean incrementing = false;
                        while (running) {
                            ri = (ri == rank) ? 0 : ri;
                            if (!incrementing) {
                                boolean isMatch = true;
                                for (int rii = 0; rii < rank; rii++) {
                                    isMatch = (t1Idx[rii] < t1Shp[rii] && t1Idx[rii] >= 0) && isMatch;
                                }
                                if (isMatch) {
                                    value += operation.execute(t0Idx, t1Idx, t2Idx);
                                }
                                incrementing = true;
                                ri = 0;
                            } else {//incrementing:
                                if (t2Idx[ri] < t2Shp[ri]) {
                                    t2Idx[ri]++;
                                    if (t2Idx[ri] == t2Shp[ri]) {
                                        running = running && !(ri == (rank - 1));
                                        t1Idx[ri] = t0Idx[ri];
                                        t2Idx[ri] = 0;
                                        ri++;
                                    } else {
                                        if (t0Shp[ri] > t1Shp[ri]) {
                                            t1Idx[ri] = (t0Idx[ri] - t2Idx[ri]);
                                        } else {
                                            t1Idx[ri] = (t0Idx[ri] + t2Idx[ri]);
                                        }
                                        incrementing = false;
                                    }
                                } else {
                                    ri++;
                                }
                            }
                        }
                        //set value in drn:
                        t0_value[t0_drn.i_of_idx(t0Idx)] = value;
                        //increment on drain:
                        Tsr.Utility.Indexing.increment(t0Idx, t0Shp);
                        i++;
                    }
                }

            }

            @Contract(pure = true)
            public static void broadcast(
                    Tsr t0_drn, Tsr t1_src, Tsr t2_src,
                    int d,
                    int i, int end,
                    Type.Operator operation
            ) {
                int[] t0Shp = t0_drn.shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
                int[] t1Shp = t1_src.shape();
                int[] t2Shp = (t2_src != null) ? t2_src.shape() : t1Shp;
                int rank = t0Shp.length;
                int[] t0Idx = t0_drn.idx_of_i(i);//new int[rank];
                int[] t1Idx = new int[rank];
                int[] t2Idx = new int[rank];
                double[] t0_value = t0_drn.value64();
                //double[] t1_value = t1_src.value64();
                //double[] t2_value = t2_src.value64();
                //int drnSze = t0_drn.size();
                //int i = 0;
                if (d < 0) {
                    while (i < end) {//increment on drain accordingly:
                        int ri = 0;
                        while (ri < rank) {
                            if (t1Shp[ri] == t2Shp[ri]) {//Equal shapes -> out index is t1 & t2 index!for this ri
                                t1Idx[ri] = t0Idx[ri];
                                t2Idx[ri] = t0Idx[ri];
                            } else if (t1Shp[ri] > t2Shp[ri]) {//Current shape axis of t2 must be 1 !
                                t1Idx[ri] = t0Idx[ri];
                                t2Idx[ri] = 0;//...therefore it can be set to 0!
                            } else if (t1Shp[ri] < t2Shp[ri]) {//same principle:
                                t1Idx[ri] = 0;
                                t2Idx[ri] = t0Idx[ri];
                            }
                            ri++;
                        }
                        //----------
                        //setInto _value in drn:
                        t0_value[t0_drn.i_of_idx(t0Idx)] = operation.execute(t0Idx, t1Idx, t2Idx);

                        //increment on drain:
                        Tsr.Utility.Indexing.increment(t0Idx, t0Shp);
                        i++;
                    }
                }
                else//---//Note: src2 is now former drain!
                {
                    while (i < end) {//increment on drain accordingly:
                        int ri = 0;
                        while (ri < rank) {
                            if (t0Shp[ri] == t1Shp[ri]) {
                                t1Idx[ri] = t0Idx[ri];//all shapes are equal -> shape index can be inherited from origin!
                                t2Idx[ri] = t0Idx[ri];
                            } else if (t0Shp[ri] > t1Shp[ri]) {
                                t1Idx[ri] = 0;//Current origin index is larger: index can be inherited!
                                t2Idx[ri] = t0Idx[ri];
                            }
                            ri++;
                        }
                        //----------
                        // multiplication:
                        double value = 0;
                        boolean running = true;
                        boolean incrementing = false;
                        while (running) {
                            ri = (ri == rank) ? 0 : ri;
                            if (!incrementing) {
                                value += operation.execute(t0Idx, t1Idx, t2Idx);
                                incrementing = true;
                                ri = 0;
                            } else {//incrementing:
                                if (t0Shp[ri] < t1Shp[ri]) {//Only if origin shape is smaller than handle and drain!
                                    t1Idx[ri]++;
                                    t2Idx[ri]++;
                                    if (t1Idx[ri] == t1Shp[ri]) {
                                        t1Idx[ri] = 0;
                                        t2Idx[ri] = 0;
                                        running = running && !(ri == (rank - 1));
                                        ri++;
                                    } else {
                                        incrementing = false;//return to calculation!
                                    }
                                } else {
                                    running = running && !(ri == (rank - 1));
                                    ri++;
                                }
                            }
                        }
                        //set value in drn:
                        t0_value[t0_drn.i_of_idx(t0Idx)] = value;
                        //increment on drain:
                        Tsr.Utility.Indexing.increment(t0Idx, t0Shp);
                        i++;
                    }
                }
            }

            @Contract(pure = true)
            private static void activate(
                    Tsr t0_drn, int i, int end, Type.Operator operation
            ) {
                int[] t0Shp = t0_drn.shape();//Tsr t0_origin, Tsr t1_handle, Tsr t2_drain ... when d>=0
                int rank = t0Shp.length;
                int[] t0Idx = t0_drn.idx_of_i(i);//new int[rank];
                int[] t1Idx = new int[rank];
                double[] t0_value = t0_drn.value64();
                while (i < end) {//increment on drain accordingly:
                    int ri = 0;
                    while (ri < rank) {
                        t1Idx[ri] = t0Idx[ri];
                        ri++;
                    }
                    //setInto _value in drn:
                    t0_value[t0_drn.i_of_idx(t0Idx)] = operation.execute(t0Idx, t1Idx, null);
                    //increment on drain:
                    Tsr.Utility.Indexing.increment(t0Idx, t0Shp);
                    i++;
                }

            }



        }


    }


}
