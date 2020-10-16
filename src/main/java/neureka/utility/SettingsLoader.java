package neureka.utility;

import groovy.lang.Closure;
import groovy.lang.GroovyShell;
import groovy.lang.GroovySystem;
import neureka.Neureka;

/**
 *  This class is a helper class for Neureka instances.
 *  It tries to execute groovy scripts used as
 *  settings for said instances.
 *
 *  This logic is not included inside the Neureka class
 *  itself because otherwise there would be an obligate dependency
 *  on groovy.
 *  If groovy lang dependencies are however not found, then
 *  this very class will not be used, initialized and therefore
 *  neureka will continue to work without groovy.
 */
public class SettingsLoader
{
    private static String _settings_source;
    private static String _setup_source;

    public static Object tryGroovyClosureOn(Object closure, Object delegate) {
            ( (Closure) closure ).setDelegate(delegate);
            return ( (Closure) closure ).call(delegate);
    }

    public static void tryGroovyScriptsOn(Neureka instance)
    {
        if (_settings_source == null || _setup_source == null) {
            _settings_source = instance.utility().readResource("library_settings.groovy");
            _setup_source = instance.utility().readResource("scripting_setup.groovy");
        }
        try {

            String version = GroovySystem.getVersion();
            if(Integer.parseInt(version.split("\\.")[ 0 ]) < 3) {
                throw new IllegalCallerException(
                        "Wrong groovy version "+version+" found! Version 3.0.0 or greater required."
                );
            }
            new GroovyShell(instance.getClass().getClassLoader()).evaluate(_settings_source);
            new GroovyShell(instance.getClass().getClassLoader()).evaluate(_setup_source);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
