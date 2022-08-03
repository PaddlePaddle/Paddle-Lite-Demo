package com.baidu.paddle.lite.demo.object_detection;

import com.baidu.paddle.lite.demo.object_detection.slice.MainAbilitySlice;
import ohos.aafwk.ability.Ability;
import ohos.aafwk.content.Intent;
import ohos.agp.window.service.WindowManager;

public class MainAbility extends Ability {
    @Override
    protected void onStart(Intent intent) {
        super.onStart(intent);
        super.setMainRoute(MainAbilitySlice.class.getName());
        getWindow().addFlags(WindowManager.LayoutConfig.MARK_SCREEN_ON_ALWAYS);
    }
}
