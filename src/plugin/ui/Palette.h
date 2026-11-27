#pragma once

#include "core/Types.h"

#include <juce_gui_basics/juce_gui_basics.h>

namespace svdb::ui
{

// Barebones dark studio palette matching FL Studio's dark chrome.
namespace col
{
inline const juce::Colour bg          { 0xff141619 };
inline const juce::Colour panel       { 0xff1b1e23 };
inline const juce::Colour rowAlt      { 0xff181b20 };
inline const juce::Colour rowSelected { 0xff283242 };
inline const juce::Colour border      { 0xff2b3039 };
inline const juce::Colour controlBg   { 0xff21252c };
inline const juce::Colour controlOn   { 0xff36475e };
inline const juce::Colour text        { 0xffe2e6ec };
inline const juce::Colour muted       { 0xff838b97 };
inline const juce::Colour accent      { 0xff4ea1ff };
inline const juce::Colour playhead    { 0xffff9f43 };
inline const juce::Colour waveform    { 0xff6e7b8f };
} // namespace col

inline juce::Colour categoryColour (Category c)
{
    // Refined XLN XO-inspired high-clarity studio palette (17 categories).
    switch (c)
    {
        case Category::Kick:      return juce::Colour (0xffdf4258); // crimson red
        case Category::Snare:     return juce::Colour (0xff2e9cf0); // clear blue
        case Category::Rim:       return juce::Colour (0xff67c0ff); // sky blue
        case Category::Clap:      return juce::Colour (0xffeb862d); // warm amber-orange
        case Category::ClosedHat: return juce::Colour (0xffe5cc38); // crisp gold
        case Category::OpenHat:   return juce::Colour (0xffb6db3b); // yellow-lime
        case Category::Cymbal:    return juce::Colour (0xff34c46a); // clean emerald
        case Category::Tom:       return juce::Colour (0xffe0684c); // coral terracotta
        case Category::Perc:      return juce::Colour (0xff2ec4b6); // turquoise-teal
        case Category::Bass808:   return juce::Colour (0xffa259e6); // vivid violet
        case Category::Bass:      return juce::Colour (0xff7850cf); // deep indigo-violet
        case Category::Synth:     return juce::Colour (0xff5276f8); // cobalt blue
        case Category::Chord:     return juce::Colour (0xff8294ff); // soft periwinkle
        case Category::Stab:      return juce::Colour (0xff4bc8ff); // electric sky
        case Category::Vocal:     return juce::Colour (0xffeb67a8); // rose pink
        case Category::Foley:     return juce::Colour (0xff82b872); // organic sage
        case Category::Riser:     return juce::Colour (0xffd946c8); // electric magenta
        case Category::Impact:    return juce::Colour (0xffc93878); // deep ruby-magenta
        case Category::Other:
        default:                  return juce::Colour (0xff7b8494); // muted slate
    }
}

} // namespace svdb::ui
