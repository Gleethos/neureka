package neureka;

import groovy.lang.Closure;
import groovy.lang.GroovyShell;
import groovy.lang.GroovySystem;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class Neureka
{
    private static final Map<Thread, Neureka> _instances;
    private static String _settings_source;
    private static String _setup_source;
    private static String _version;

    private static boolean _groovyAvailable;

    private Settings _settings;
    private Utility _utility;

    static {
        _instances = new ConcurrentHashMap<>();
        _groovyAvailable = Utility.isPresent("groovy.lang.GroovySystem");
    }

    private Neureka(){
        _settings = new Settings();
        _utility = new Utility();
    }

    public static Neureka instance(){
        return instance(Thread.currentThread());
    }

    public static void setContext(Thread thread, Neureka instance){
        _instances.put(thread, instance);
    }

    public static Neureka instance(Thread thread){
        if(_instances.containsKey(thread)) return _instances.get(thread);
        else {
            Neureka instance = new Neureka();
            setContext(thread, instance);
            synchronized (Neureka.class) {
                instance.reset();
            }
            return instance;
        }
    }

    public static Neureka instance(Object closure) {
        Object o = Utility.tryGroovyClosureOn(closure, Neureka.instance());
        if (o instanceof String) _version = (String) o;
        return Neureka.instance();
    }

    public Settings settings(){
        return _settings;
    }

    public Settings settings(Object closure){
        Utility.tryGroovyClosureOn(closure, _settings);
        return _settings;
    }

    public Utility utility(){
        return _utility;
    }

    public static String version(){
        return _version;
    }

    public void reset() {
        if (_groovyAvailable) {
            if (_settings_source == null || _setup_source == null) {
                _settings_source = utility().readResource("library_settings.groovy");
                _setup_source = utility().readResource("scripting_setup.groovy");
            }
            try {
                Utility.tryGroovyScriptsOn(this);
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            settings().autoDiff().setIsRetainingPendingErrorForJITProp(true);
            settings().autoDiff().setIsApplyingGradientWhenTensorIsUsed(true);
            settings().autoDiff().setIsApplyingGradientWhenRequested(true);
            settings().indexing().setIsUsingLegacyIndexing(false);
            settings().indexing().setIsUsingThoroughIndexing(true);
            settings().debug().setIsKeepingDerivativeTargetPayloads(false);
            settings().view().setIsUsingLegacyView(false);
        }
    }

    private boolean _currentThreadIsAuthorized(){
        return this.equals(_instances.get(Thread.currentThread()));
    }

    public class Settings
    {
        private Debug _debug;
        private AutoDiff _autoDiff;
        private Indexing _indexing;
        private View _view;
        private NDim _ndim;

        private boolean _isLocked = false;

        private Settings() {
            _debug = new Debug();
            _autoDiff = new AutoDiff();
            _indexing = new Indexing();
            _view = new View();
            _ndim = new NDim();
        }

        public Debug debug() {
            return _debug;
        }

        public Debug debug(Object closure) {
            Utility.tryGroovyClosureOn(closure, _debug);
            return _debug;
        }

        public AutoDiff autoDiff(){
            return _autoDiff;
        }

        public AutoDiff autoDiff(Object closure) {
            Utility.tryGroovyClosureOn(closure, _autoDiff);
            return _autoDiff;
        }

        public Indexing indexing(){
            return _indexing;
        }

        public Indexing indexing(Object closure) {
            Utility.tryGroovyClosureOn(closure, _indexing);
            return _indexing;
        }

        public View view(){
            return _view;
        }

        public View view(Object closure) {
            Utility.tryGroovyClosureOn(closure, _view);
            return _view;
        }

        public NDim ndim(){
            return _ndim;
        }

        public NDim ndim(Object closure) {
            Utility.tryGroovyClosureOn(closure, _ndim);
            return _ndim;
        }

        public boolean isLocked(){
            return  _isLocked;
        }

        public void setIsLocked(boolean locked) {
            _isLocked = locked;
        }

        public class Debug
        {
            /**
             * Every derivative is calculated with respect to some graph node.
             * Graph nodes contain payload tensors.
             * A tensor might not always be used for backpropagation.
             * Therefore it will be deleted if possible.
             * Targeted tensors are either leave tensors (They require gradients)
             * or they are angle points between forward- and reverse-mode-AutoDiff!
             * In this case:
             * If the tensor is not needed for backpropagation it will be deleted.
             * The graph node will dereference the tensor either way.
             *
             * The flag determines this behavior with respect to target nodes.
             * It is used in the test suit to validate that the right tensors were calculated.
             * This flag should not be modified in production! (memory leak)
             */
            private boolean _isKeepingDerivativeTargetPayloads = false;

            public boolean isKeepingDerivativeTargetPayloads(){
                return _isKeepingDerivativeTargetPayloads;
            }

            public void setIsKeepingDerivativeTargetPayloads(boolean keep){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isKeepingDerivativeTargetPayloads = keep;
            }

        }

        public class AutoDiff // Auto-Differentiation
        {
            /**
             * This flag enables an optimization technique which only applies
             * gradients as soon as they are needed by a tensor (the tensor is used again).
             * If the flag is set to true
             * then error values will accumulate whenever it makes sense.
             * This technique however uses more memory but will
             * improve performance for some networks substantially.
             * The technique is termed JIT-Propagation.
             */
            private boolean _isRetainingPendingErrorForJITProp = true;

            /**
             * Gradients will automatically be applied to tensors as soon as
             * they are being used for calculation (GraphNode instantiation).
             * This feature works well with JIT-Propagation.
             */
            private boolean _isApplyingGradientWhenTensorIsUsed = true;

            /**
             * Gradients will only be applied if requested.
             * Usually this happens immediately, however
             * if the flag 'applyGradientWhenTensorIsUsed' is set
             * to true, then the tensor will only be updated by its
             * gradient if requested AND tensor is used! (GraphNode instantiation).
             */
            private boolean _isApplyingGradientWhenRequested = true;

            public boolean isRetainingPendingErrorForJITProp(){
                return _isRetainingPendingErrorForJITProp;
            }

            public void setIsRetainingPendingErrorForJITProp(boolean retain){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isRetainingPendingErrorForJITProp = retain;
            }

            public boolean isApplyingGradientWhenTensorIsUsed(){
                return _isApplyingGradientWhenTensorIsUsed;
            }

            public void setIsApplyingGradientWhenTensorIsUsed(boolean apply){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isApplyingGradientWhenTensorIsUsed = apply;
            }

            public boolean isApplyingGradientWhenRequested(){
                return _isApplyingGradientWhenRequested;
            }

            public void setIsApplyingGradientWhenRequested(boolean apply) {
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isApplyingGradientWhenRequested = apply;
            }

        }

        public class Indexing
        {
            private boolean _isUsingLegacyIndexing = false;

            private boolean _isUsingThoroughIndexing;

            public boolean isUsingLegacyIndexing(){
                return _isUsingLegacyIndexing;
            }

            public void setIsUsingLegacyIndexing(boolean enabled) {
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isUsingLegacyIndexing = enabled;//NOTE: gpu code must recompiled! (in OpenCLPlatform)
            }

            public boolean isUsingThoroughIndexing(){
                return _isUsingThoroughIndexing;
            }

            public void setIsUsingThoroughIndexing(boolean thorough){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isUsingThoroughIndexing = thorough;
            }

        }

        public class View
        {
            private boolean _isUsingLegacyView = false;

            public boolean isUsingLegacyView(){
                return _isUsingLegacyView;
            }

            public void setIsUsingLegacyView(boolean enabled){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isUsingLegacyView = enabled;
            }

        }

        public class NDim
        {
            /**
             *  The DefaultNDConfiguration class stores shape, translation...
             *  as cached int arrays.
             *  Disabling this flag allows for custom 1D, 2D, 3D classes to be loaded. (Improves memory locality)
             */
            private boolean _isOnlyUsingDefaultNDConfiguration = false;

            public boolean isOnlyUsingDefaultNDConfiguration(){
                return _isOnlyUsingDefaultNDConfiguration;
            }

            public void setIsOnlyUsingDefaultNDConfiguration(boolean enabled){
                if(_isLocked || !_currentThreadIsAuthorized()) return;
                _isOnlyUsingDefaultNDConfiguration = enabled;
            }

        }

    }

    public static class Utility {
        /**
         * Helper method which reads the file with the given name and returns
         * the contents of this file as a String. Will exit the application
         * if the file can not be read.
         *
         * @param path
         * @return The contents of the file
         */
        public String readResource(String path){
            InputStream stream = getClass().getClassLoader().getResourceAsStream(path);
            try {
                BufferedReader br = new BufferedReader(new InputStreamReader(stream));
                StringBuffer sb = new StringBuffer();
                String line = "";
                while (line!=null) {
                    line = br.readLine();
                    if (line != null) sb.append(line).append("\n");
                }
                return sb.toString();
            } catch (IOException e) {
                e.printStackTrace();
                System.exit(1);
                return null;
            }
        }

        public static Object tryGroovyClosureOn(Object closure, Object delegate) {
            if (_groovyAvailable) {
                ((Closure) closure).setDelegate(delegate);
                return ((Closure) closure).call(delegate);
            }
            return null;
        }

        public static void tryGroovyScriptsOn(Neureka instance) {
            if(_groovyAvailable) {
                String version = GroovySystem.getVersion();
                if(Integer.parseInt(version.split("\\.")[0]) < 3) {
                    throw new IllegalCallerException(
                            "Wrong groovy version "+version+" found! Version 3.0.0 or greater required."
                    );
                }
                new GroovyShell(instance.getClass().getClassLoader()).evaluate(_settings_source);
                new GroovyShell(instance.getClass().getClassLoader()).evaluate(_setup_source);
            }
        }

        public static boolean isPresent(String className){
            boolean found = false;
            try {
                Class.forName(className);
                found = true;
            } catch (Throwable ex) {// Class or one of its dependencies is not present...
                System.out.println("[Info]: Groovy dependencies not found! Neureka settings uninitialized!\n[Cause]: "+ex.getMessage());
                return found;
            } finally {
                if(!found) System.out.println("[Info]: Groovy dependencies not found! Neureka settings uninitialized!");
                return found;
            }
        }

    }


}
