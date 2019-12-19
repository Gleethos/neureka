package neureka.acceleration;

import neureka.Tsr;
import neureka.function.Function;
import org.jetbrains.annotations.Contract;

import java.util.Collection;

public class CPU extends AbstractDevice {
    public CPU() {
    }

    @Override
    protected void _enqueue(Tsr[] tsrs, int d, int f_id) {
        for(Tsr t : tsrs) t.setIsVirtual(false);
        switch (Function.TYPES.REGISTER[f_id]) {
            case "sig": exec.activate_sigmoid(tsrs[0], tsrs[1], d); break;
            case "sin": exec.activate_sinus(tsrs[0], tsrs[1], d);break;
            case "cos": exec.activate_cosinus(tsrs[0], tsrs[1], d);break;
            case "abs": exec.activate_absolute(tsrs[0], tsrs[1], d);break;
            case "lig": exec.activate_ligmoid(tsrs[0], tsrs[1], d);break;
            case "tanh": exec.activate_tanh(tsrs[0], tsrs[1], d);break;
            case "gaus": exec.activate_gaussian(tsrs[0], tsrs[1], d);break;
            case "quad": exec.activate_quadratic(tsrs[0], tsrs[1], d);break;
            case "idy": exec.activate_identity(tsrs[0], tsrs[1], d);break;
            case "relu": exec.activate_relu(tsrs[0], tsrs[1], d);break;
            case "sum": exec.broadcast_add(tsrs[0], tsrs[1], tsrs[2], d);break;
            case "prod": exec.broadcast_multiply(tsrs[0], tsrs[1], tsrs[2], d);break;
            //---
            case "x":
                if (d >= 0) {
                    if (d == 0) tsrs[0] = tsrs[2]; else tsrs[0] = tsrs[1];
                } else {
                    exec.convolve_multiply(tsrs[0], tsrs[1], tsrs[2]);
                }
                break;
            case ("x" + ((char) 187)): exec.convolve_multiply_inverse(tsrs[2], tsrs[1], tsrs[0]);break;
            case ("" + ((char) 171)) + "x": exec.convolve_multiply_inverse(tsrs[0], tsrs[1], tsrs[2]);break;
            //---
            case "a": exec.convolve_add(tsrs[0], tsrs[1], tsrs[2], d);break;
            case ("a" + ((char) 187)): exec.convolve_add_inverse(tsrs[2], tsrs[1], tsrs[0]);break;
            case ("" + ((char) 171)) + "a": exec.convolve_add_inverse(tsrs[0], tsrs[1], tsrs[2]);break;
            //---
            case "s": exec.convolve_subtract(tsrs[0], tsrs[1], tsrs[2], d);break;
            case ("s" + ((char) 187)): exec.convolve_subtract_inverse(tsrs[2], tsrs[1], tsrs[0]);break;
            case ("" + ((char) 171)) + "s": exec.convolve_subtract_inverse(tsrs[0], tsrs[1], tsrs[2]);break;
            //---
            case "d": exec.convolve_divide(tsrs[0], tsrs[1], tsrs[2], d);break;
            case ("d" + ((char) 187)):
                //exec.convolve_divide_inverse(tsrs[2], tsrs[1], tsrs[0]);
                break;
            case ("" + ((char) 171)) + "d":
                //exec.convolve_divide_inverse(tsrs[0], tsrs[1], tsrs[2]);
                break;
            //---
            case "p":
                exec.convolve_power(tsrs[0], tsrs[1], tsrs[2], d);
                break;
            case ("p" + ((char) 187)):
                //exec.convolve_power_inverse(tsrs[2], tsrs[1], tsrs[0]);
                break;
            case ("" + ((char) 171)) + "p":
                //exec.convolve_power_inverse(tsrs[0], tsrs[1], tsrs[2]);
                break;
            //---
            case "m":
                exec.convolve_mod(tsrs[0], tsrs[1], tsrs[2], d);
                break;
            case ("m" + ((char) 187)):
                //exec.convolve_mod_inverse(tsrs[2], tsrs[1], tsrs[0]);
                break;
            case ("" + ((char) 171)) + "m":
                //exec.convolve_mod_inverse(tsrs[0], tsrs[1], tsrs[2]);
                break;
            //---

            case "*": exec.broadcast_multiply(tsrs[0], tsrs[1], tsrs[2], d);break;
            case "+": exec.broadcast_add(tsrs[0], tsrs[1], tsrs[2], d);break;
            case "-": exec.broadcast_subtract(tsrs[0], tsrs[1], tsrs[2], d);break;
            case "/": exec.broadcast_divide(tsrs[0], tsrs[1], tsrs[2], d);break;
            case "%": exec.broadcast_mod(tsrs[0], tsrs[1], tsrs[2], d);break;
            case "^": exec.broadcast_power(tsrs[0], tsrs[1], tsrs[2], d);break;
            case "<": exec.activate_identity(tsrs[0], tsrs[1], d);break;
            case ">": exec.activate_identity(tsrs[1], tsrs[0], d);break;
            default:
                throw new IllegalStateException("[CPU][enqueue]: Operation not found!");
        }
    }

    @Override
    protected void _enqueue(Tsr t, double value, int d, int f_id) {
        int[] shape = new int[t.rank()];
        for (int i = 0; i < shape.length; i++) shape[i] = 1;
        _enqueue(new Tsr[]{t, t, new Tsr(shape, value)}, d, f_id);
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

    public static class exec {
        interface Range {
            void execute(int start, int end);
        }

        interface Operator {
            double execute(int[] t0Idx, int[] t1Idx, int[] t2Idx);
        }


        private static int _adjusted_d(int d, Tsr t0_drn, Tsr t1_src, Tsr t2_src) {
            for (int i = 0; i < t0_drn.rank(); i++)
                d = (t0_drn.shape(i) != t1_src.shape(i) || t1_src.shape(i) != t2_src.shape(i)) ? -1 : d;
            return d;
        }

        //---

        public static void activate_sigmoid(
                Tsr t0_drn,
                Tsr t1_src,
                int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.activate(
                        t0_drn,// t1_src, null, d,
                        start, end,
                        _sigmoid(t1_src, d)
                );
            });
        }

        private static Operator _sigmoid(
                Tsr t1_src,
                int d
        ) {
            double[] t1_val = t1_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) ->
                        1 / (1 + Math.pow(Math.E, -t1_val[t1_src.i_of_idx(t1Idx)]));
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return (1 - Math.pow(((input) / Math.pow((1 + Math.pow((input), 2)), 0.5)), 2));
                };
            }
        }

        //---

        public static void activate_tanh(
                Tsr t0_drn,
                Tsr t1_src,
                int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.activate(
                        t0_drn,// t1_src, null, d,
                        start, end,
                        _tanh(t1_src, d)
                );
            });
        }

        private static Operator _tanh(
                Tsr t1_src,
                int d
        ) {
            double[] t1_val = t1_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return ((input)) / Math.pow((1 + Math.pow(((input)), 2)), 0.5);
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return (1 - Math.pow(((input) / Math.pow((1 + Math.pow((input), 2)), 0.5)), 2));
                };
            }
        }

        //---

        public static void activate_quadratic(
                Tsr t0_drn,
                Tsr t1_src,
                int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.activate(
                        t0_drn,// t1_src, null, d,
                        start, end,
                        _quadratic(t1_src, d)
                );
            });
        }

        private static Operator _quadratic(
                Tsr t1_src,
                int d
        ) {
            double[] t1_val = t1_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return ((input) * (input));
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) ->
                        2 * t1_val[t1_src.i_of_idx(t1Idx)];
            }
        }

        //---

        public static void activate_ligmoid(
                Tsr t0_drn,
                Tsr t1_src,
                int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.activate(
                        t0_drn,
                        start, end,
                        _ligmoid(t1_src, d)
                );
            });
        }

        private static Operator _ligmoid(
                Tsr t1_src,
                int d
        ) {
            double[] t1_val = t1_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return (Math.log(1 + Math.pow(Math.E, input)));
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> _sigmoid(t1_src, -1).execute(t0Idx, t1Idx, t2Idx);
            }
        }

        //---

        public static void activate_gaussian(
                Tsr t0_drn,
                Tsr t1_src,
                int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.activate(
                        t0_drn,
                        start, end,
                        _gaussian(t1_src, d)
                );
            });
        }

        private static Operator _gaussian(
                Tsr t1_src,
                int d
        ) {
            double[] t1_val = t1_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return Math.pow(Math.E, -Math.pow((input), 2));
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return -2 * ((input)) * Math.pow(Math.E, -Math.pow((input), 2));
                };

            }
        }

        //---

        public static void activate_absolute(
                Tsr t0_drn,
                Tsr t1_src,
                int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.activate(
                        t0_drn,
                        start, end,
                        _absolute(t1_src, d)
                );
            });
        }

        private static Operator _absolute(
                Tsr t1_src,
                int d
        ) {
            double[] t1_val = t1_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return Math.abs(input);
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return (input < 0) ? -1 : 1;
                };

            }
        }

        //---

        public static void activate_sinus(
                Tsr t0_drn,
                Tsr t1_src,
                int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.activate(
                        t0_drn,
                        start, end,
                        _sinus(t1_src, d)
                );
            });
        }

        private static Operator _sinus(
                Tsr t1_src,
                int d
        ) {
            double[] t1_val = t1_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return Math.sin(input);
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return Math.cos(input);
                };

            }
        }

        //---

        public static void activate_cosinus(
                Tsr t0_drn,
                Tsr t1_src,
                int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.activate(
                        t0_drn,
                        start, end,
                        _cosinus(t1_src, d)
                );
            });
        }

        private static Operator _cosinus(
                Tsr t1_src,
                int d
        ) {
            double[] t1_val = t1_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return Math.cos(input);
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    double input = t1_val[t1_src.i_of_idx(t1Idx)];
                    return -Math.sin(input);
                };

            }
        }

        //---

        //---

        public static void activate_relu(
                Tsr t0_drn,
                Tsr t1_src,
                int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.activate(
                        t0_drn,// t1_src, null, d,
                        start, end,
                        _relu(t1_src, d)
                );
            });
        }

        private static Operator _relu(
                Tsr t1_src, int d
        ) {
            double[] t1_val = t1_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    if(t1_val[t1_src.i_of_idx(t1Idx)]>=0){
                        return t1_val[t1_src.i_of_idx(t1Idx)];
                    }
                    return t1_val[t1_src.i_of_idx(t1Idx)]*0.01;
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    if(t1_val[t1_src.i_of_idx(t1Idx)]>=0){
                        return 1;
                    }
                    return 0.01;
                };
            }
        }


        //---

        public static void activate_identity(
                Tsr t0_drn,
                Tsr t1_src,
                int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.activate(
                        t0_drn,// t1_src, null, d,
                        start, end,
                        _identity(t1_src, d)
                );
            });
        }

        private static Operator _identity(
                Tsr t1_src,
                int d
        ) {
            double[] t1_val = t1_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    return t1_val[t1_src.i_of_idx(t1Idx)];
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    return t1_val[t1_src.i_of_idx(t1Idx)];
                };
            }
        }


        //---
        public static void convolve_multiply(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src
        ) {
            _threaded(t0_drn.size(), ((start, end) -> {
                _template.convolve(
                        t0_drn, t1_src, t2_src, -1,
                        start, end,
                        _multiplication(t1_src, t2_src, -1)
                );
            }));
        }
        public static void convolve_multiply_inverse(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src
        ) {
            _threaded(t0_drn.size(), ((start, end) -> {
                _template.convolve(
                        t0_drn, t1_src, t2_src, 0,
                        start, end,
                        _multiplication(t1_src, t2_src, -1)
                );
            }));
        }

        public static void broadcast_multiply(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.broadcast(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _multiplication(t1_src, t2_src, _adjusted_d(d, t0_drn, t1_src, t2_src))//if adjusted throw exception!
                );
            });
        }
        public static void broadcast_multiply_inverse(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.broadcast(
                        t0_drn, t1_src, t2_src, 0,
                        start, end,
                        _multiplication(t1_src, t2_src, -1)//if adjusted throw exception!
                );
            });
        }


        private static Operator _multiplication(
                Tsr t1_src, Tsr t2_src, int d
        ) {
            double[] t1_val = t1_src.value64();
            double[] t2_val = t2_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    return t1_val[t1_src.i_of_idx(t1Idx)] * t2_val[t2_src.i_of_idx(t2Idx)];
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    if (d == 0) {
                        return t2_val[t2_src.i_of_idx(t2Idx)];
                    } else {
                        return t1_val[t1_src.i_of_idx(t1Idx)];
                    }

                };
            }
        }

        //---

        public static void convolve_add(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), ((start, end) -> {
                _template.convolve(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _addition(t1_src, t2_src, d)
                );
            }));
        }
        public static void convolve_add_inverse(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src
        ) {
            _threaded(t0_drn.size(), ((start, end) -> {
                _template.convolve(
                        t0_drn, t1_src, t2_src, 0,
                        start, end,
                        _addition(t1_src, t2_src, -1)
                );
            }));
        }

        public static void broadcast_add(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.broadcast(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _addition(t1_src, t2_src, d)
                );
            });
        }
        public static void broadcast_add_inverse(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.broadcast(
                        t0_drn, t1_src, t2_src, 0,
                        start, end,
                        _addition(t1_src, t2_src, 0)
                );
            });
        }

        private static Operator _addition(
                Tsr t1_src, Tsr t2_src, int d
        ) {
            double[] t1_val = t1_src.value64();
            double[] t2_val = t2_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    return t1_val[t1_src.i_of_idx(t1Idx)] + t2_val[t2_src.i_of_idx(t2Idx)];
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> 1.0;
            }
        }

        //---

        public static void convolve_subtract(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), ((start, end) -> {
                _template.convolve(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _subtraction(t1_src, t2_src, d)
                );
            }));
        }
        public static void convolve_subtract_inverse(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src
        ) {
            _threaded(t0_drn.size(), ((start, end) -> {
                _template.convolve(
                        t0_drn, t1_src, t2_src, 0,
                        start, end,
                        _subtraction(t1_src, t2_src, -1)
                );
            }));
        }

        public static void broadcast_subtract(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.broadcast(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _subtraction(t1_src, t2_src, d)
                );
            });
        }
        public static void broadcast_subtract_inverse(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.broadcast(
                        t0_drn, t1_src, t2_src, 0,
                        start, end,
                        _subtraction(t1_src, t2_src, -1)
                );
            });
        }

        private static Operator _subtraction(
                Tsr t1_src, Tsr t2_src, int d
        ) {
            double[] t1_val = t1_src.value64();
            double[] t2_val = t2_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    return t1_val[t1_src.i_of_idx(t1Idx)] - t2_val[t2_src.i_of_idx(t2Idx)];
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    return (d == 0) ? 1.0 : -1.0;
                };
            }
        }

        //---

        public static void convolve_divide(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), ((start, end) -> {
                _template.convolve(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _division(t1_src, t2_src, d)
                );
            }));
        }

        public static void broadcast_divide(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.broadcast(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _division(t1_src, t2_src, d)
                );
            });
        }

        private static Operator _division(
                Tsr t1_src, Tsr t2_src, int d
        ) {
            double[] t1_val = t1_src.value64();
            double[] t2_val = t2_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    return t1_val[t1_src.i_of_idx(t1Idx)] / t2_val[t2_src.i_of_idx(t2Idx)];
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    if (d == 0) {
                        return 1 / t2_val[t2_src.i_of_idx(t2Idx)];
                    } else {
                        return
                                -(t1_val[t1_src.i_of_idx(t1Idx)]
                                        /
                                        Math.pow(t2_val[t2_src.i_of_idx(t2Idx)], 2));
                    }
                };
            }
        }
        //---

        //---
        public static void convolve_power(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), ((start, end) -> {
                _template.convolve(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _power(t1_src, t2_src, d)
                );
            }));
        }

        public static void broadcast_power(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.broadcast(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _power(t1_src, t2_src, d)
                );
            });
        }

        private static Operator _power(
                Tsr t1_src, Tsr t2_src, int d
        ) {
            double[] t1_val = t1_src.value64();
            double[] t2_val = t2_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    return Math.pow(
                            t1_val[t1_src.i_of_idx(t1Idx)],
                            t2_val[t2_src.i_of_idx(t2Idx)]
                    );
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    if (d == 0) {
                        return t2_val[t2_src.i_of_idx(t2Idx)]
                                * Math.pow(
                                t1_val[t1_src.i_of_idx(t1Idx)],
                                t2_val[t2_src.i_of_idx(t2Idx)] - 1
                        );
                    } else {
                        return Math.pow(
                                t1_val[t1_src.i_of_idx(t1Idx)],
                                t2_val[t2_src.i_of_idx(t2Idx)]
                        ) * Math.log(t1_val[t1_src.i_of_idx(t1Idx)]);
                    }
                };
            }
        }

        //---
        //---
        public static void convolve_mod(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), ((start, end) -> {
                _template.convolve(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _modulo(t1_src, t2_src, d)
                );
            }));
        }

        public static void broadcast_mod(
                Tsr t0_drn, Tsr t1_src, Tsr t2_src, int d
        ) {
            _threaded(t0_drn.size(), (start, end) -> {
                _template.broadcast(
                        t0_drn, t1_src, t2_src, d,
                        start, end,
                        _modulo(t1_src, t2_src, d)
                );
            });
        }

        private static Operator _modulo(
                Tsr t1_src, Tsr t2_src, int d
        ) {
            double[] t1_val = t1_src.value64();
            double[] t2_val = t2_src.value64();
            if (d < 0) {
                return (t0Idx, t1Idx, t2Idx) -> {
                    return t1_val[t1_src.i_of_idx(t1Idx)] % t2_val[t2_src.i_of_idx(t2Idx)];
                };
            } else {
                return (t0Idx, t1Idx, t2Idx) -> {
                    if (d == 0) {
                        return 1 / t2_val[t2_src.i_of_idx(t2Idx)];
                    } else {
                        return
                                -(t1_val[t1_src.i_of_idx(t1Idx)]
                                        /
                                        Math.pow(t2_val[t2_src.i_of_idx(t2Idx)], 2));
                    }
                };
            }
        }
        //---

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
                    th[i] = new Thread(() -> {
                        range.execute(start, end);
                    });
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
                    Operator operation
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
                        Tsr.fcn.indexing.increment(t0Idx, t0Shp);
                        i++;
                    }
                } else//---
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
                        Tsr.fcn.indexing.increment(t0Idx, t0Shp);
                        i++;
                    }
                }

            }

            @Contract(pure = true)
            public static void broadcast(
                    Tsr t0_drn, Tsr t1_src, Tsr t2_src,
                    int d,
                    int i, int end,
                    Operator operation
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
                        t0_value[t0_drn.i_of_idx(t0Idx)] =
                                operation.execute(t0Idx, t1Idx, t2Idx);

                        //increment on drain:
                        Tsr.fcn.indexing.increment(t0Idx, t0Shp);
                        i++;
                    }
                } else//---//Note: src2 is now former drain!
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
                        Tsr.fcn.indexing.increment(t0Idx, t0Shp);
                        i++;
                    }
                }
            }

            @Contract(pure = true)
            private static void activate(
                    Tsr t0_drn, int i, int end,
                    Operator operation
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
                    Tsr.fcn.indexing.increment(t0Idx, t0Shp);
                    i++;
                }

            }


        }


    }


}
