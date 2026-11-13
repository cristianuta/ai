'use strict';
/* Pornirea aplicației. Inițializăm secvențial ca prima cerere să creeze sesiunea HTTP,
   iar a doua s-o refolosească (altfel două cereri simultane ar crea două sesiuni). */
(async function boot() {
  Tabs.show(location.hash === '#ga' ? 'ga' : 'nn');
  try {
    await NNApp.init();
    await GAApp.init();
  } catch (e) {
    toast(e.message, 'error', 8000);
  }
  Tabs.show(Tabs.active);
})();
