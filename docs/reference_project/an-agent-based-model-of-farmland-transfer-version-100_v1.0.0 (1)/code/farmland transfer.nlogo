turtles-own [
  age                      ; The age of the head of household
  off-farming              ; The proportion of
  landholding              ; The amount of land that a household holds
  family_size              ; Number of members in a household
  prob_transferIn          ; Inward transferring willingness
  prob_transferOut         ; Outward transferring willingness
  transfer-in?             ; True if prob_transferIn is above threshold_in
  transfer-out?            ; True if prob_transferOut is above threshold_out
  transfer?                ; True if ever transfer farmland in or out
]
globals [
  the_other_side           ; Agent set of transfer-out households in a transfer round
  Alandholding_t-1         ; Previous land holding for transfer-in households
  Blandholding_t-1         ; Previous land holding for transfer-out housheolds
  total-land               ; Total area of land
  sum_transfer             ; Sum are of land transferred
  Differ                   ; The increase of sum_transfer for each step
  land_concentration       ; The share of land used by the 10% households with the most land to the total amount of land
  land_transferred         ; The proportion of land transferred (in or out) to total land of the village
  land_fragmentation       ; One minus the proportion of households without land after land transfer
  stop?                    ; if True the model stops running
  sd_off_farming           ; Standard deviation of the percentages of off-farm income in the initial simulation period
  sd_age                   ; Standard deviation of the ages of household heads
  sd_size                  ; Standard deviation of the numbers of household members
  mean_size                ; Mean of the numbers of household members
  m_age                    ; Age of household head with the highest willingness to transfer land in
  n_age                    ; Age of household head with the highest willingness to transfer land out
]
to setup
  clear-all
  ;set the initial value of parameters
  set sd_off_farming 0.36
  set sd_age 11.14
  set sd_size 1.82
  set mean_size 4.09
  set m_age 45
  set n_age 30
  ;
  create-turtles number_of_households [
    set age int random-normal mean_age sd_age
    while [age < 20 or age > 80 ][set age int random-normal mean_age sd_age]
    set family_size int random-normal mean_size sd_size
    while [family_size < 1 or family_size > 10][set family_size int random-normal mean_size sd_size]
    set landholding mean_landholding * family_size
    set off-farming random-normal mean_off_farming sd_off_farming
    while [off-farming < 0 or off-farming > 1][set off-farming random-normal mean_off_farming sd_off_farming]
    set transfer? false
    set transfer-in? false
    set transfer-out? false
    set stop? false
    setxy random-xcor random-ycor
    set shape "person"
    set color white
  ]
  repeat 4 [ask turtles [create-link-with one-of other turtles ask links [set hidden? true]]] ; Create land spatial relation network
  ifelse ((2 * count links / count turtles) < average_degree) [while [(2 * count links / count turtles) < average_degree] [ask one-of turtles [create-link-with one-of other turtles]]][while [(2 * count links / count turtles) > average_degree] [ask one-of links [die]]]
  ask links [set hidden? true]
  set Differ []
  set total-land sum [ landholding ] of turtles
  reset-ticks
  update
end
to go
   if (ticks > 1) [if item 0 Differ = 0 and item 1 Differ = 0 [set stop? true stop ]]
   if (ticks = 20) [set stop? true stop ]
   transfer
   tick
   update
end
to transfer
  let prisum sum_transfer
  let A turtles with [prob_transferIn > threshold_in]
  let B turtles with [ (( landholding > 0 and age <= 50 ) or ( landholding > 0.5 * family_size and age > 50 )) and prob_transferOut > threshold_out]
  ask A  [ set transfer-in? true  ] ; Determine the agenset of household transferring land in
  ask B with [transfer-in? = false] [ set transfer-out? true ] ; Determine the agenset of household transferring land out
  set the_other_side [who] of (turtles with [transfer-out? = true])
  if (count A != 0 and length the_other_side != 0)
  [ask A [
    set Alandholding_t-1 landholding
    let i 0
    let n length the_other_side
    while [landholding < optimal_scale and i < n]
     [let current item i the_other_side
        if (in-link-neighbor? turtle current and [landholding] of turtle current > 0)[
       ask turtle current [ set Blandholding_t-1 landholding ]
       let neighbor [link-neighbors] of turtle current

       if ([age] of (turtle current) <= 55) [
         ifelse (landholding + ([landholding] of turtle current) > optimal_scale) [
            ask turtle current [set landholding landholding + [landholding] of myself - optimal_scale  set sum_transfer (sum_transfer + optimal_scale - [landholding] of myself)]
            set landholding optimal_scale
            ask n-of random count neighbor neighbor [if self != myself [create-link-with myself  ask link-with turtle current [die]]]]
          [set landholding (landholding + [landholding] of turtle current) set sum_transfer (sum_transfer + [landholding] of turtle current)
            ask turtle current [set landholding 0 ask my-links [die]] ask neighbor [if self != myself [create-link-with myself ]]]
        ] ; When age of tranfer-out household <= 55, he will transfer out all his land

          if ([age] of turtle current > 55 and [landholding] of turtle current > 0.5 * family_size) [
        ifelse (landholding + ([landholding] of turtle current) - ([family_size] of turtle current * 0.5) > optimal_scale) [
          ask turtle current [set landholding (landholding + [landholding] of myself - optimal_scale) set sum_transfer (sum_transfer + optimal_scale - [landholding] of myself)
          ask n-of random count neighbor neighbor [if self != myself [create-link-with myself  ask link-with turtle current [die]]]]
          set landholding optimal_scale]
          [set landholding (landholding + ([landholding] of turtle current) - ([family_size] of turtle current * 0.5))  set sum_transfer (sum_transfer + [landholding] of turtle current - ([family_size] of turtle current * 0.5))
          ask turtle current [set landholding ([family_size] of turtle current * 0.5)]
          ask n-of random count neighbor neighbor [if self != myself [create-link-with myself ask links [set hidden? true] ask link-with turtle current [die]]]]
        ] ; When age of transfer-out household > 55, he will keep some land for food need and transfer the rest out.

      ask turtle current [set off-farming 1 - (landholding * (1 - off-farming) / Blandholding_t-1 ) ; Update off-farming of transfer-out households
        if ( off-farming > 1 ) [ set off-farming 1 ]
        if ( off-farming < 0 ) [ set off-farming 0 ]
        set transfer? true

         ]
      ifelse ( Alandholding_t-1 = 0 ) [set off-farming random-float 1] ; Update off-farming of transfer-in households
        [set off-farming 1 - ( landholding * ( 1 - off-farming ) / Alandholding_t-1 )]
      if ( off-farming > 1 ) [ set off-farming 1 ]
      if ( off-farming < 0 ) [ set off-farming 0 ]
      set transfer-in? false
      set transfer? true
     ]
      set i i + 1
    ]
  ]]
  ask turtles [ set age age + 1 ] ; Update ages
  ask turtles [set transfer-out?  false]
  ask turtles [set transfer-in?  false]
  ask links   [set hidden? true]
  set Differ fput (sum_transfer - prisum) Differ
end
to update
  ask turtles[
    if landholding > 100 [ set off-farming 0 ]
    set prob_transferIn ( 16 * exp ( ( age - m_age ) / 10 - 5 * off-farming ) / ( ( 1 + exp ( ( age - m_age ) / 10 ) ) ^ 2 * ( 1 + exp ( - off-farming * 5  ) ) ^ 2 ) ) ; Calculate the prob_transferIn of a household
    set prob_transferOut ( 16 * exp ( ( age - n_age ) / 10 + 5 * off-farming - 5 ) / ( ( 1 + exp ( ( age - n_age ) / 10 ) ) ^ 2 * ( 1 + exp ( off-farming * 5 - 5 ) ) ^ 2 ) ) ; Calculate the prob_transferOut of a household
    set size 1 +  2 * exp(-20 / ( landholding + 0.01 ))  ;1 + landholding * number_of_household / (total-land * 10)
    if transfer? = true [set color green]
  ] ; Update the willingness to transfer
  set land_concentration sum [landholding] of max-n-of round (0.1 * number_of_households) turtles [landholding] / total-land
  set land_transferred sum [abs (landholding - mean_landholding * family_size)] of turtles / 2 / total-land
  set land_fragmentation 1 - count turtles with [landholding = 0] / number_of_households
end
@#$#@#$#@
GRAPHICS-WINDOW
370
19
962
612
-1
-1
17.7
1
10
1
1
1
0
0
0
1
-16
16
-16
16
1
1
1
ticks
30.0

BUTTON
12
69
96
106
setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
6
126
198
159
number_of_households
number_of_households
100
1000
311.0
1
1
NIL
HORIZONTAL

BUTTON
115
68
191
106
NIL
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
6
171
197
204
mean_landholding
mean_landholding
0.5
10
1.14
0.1
1
NIL
HORIZONTAL

SLIDER
7
349
199
382
optimal_scale
optimal_scale
30
120
100.0
1
1
NIL
HORIZONTAL

SLIDER
6
302
198
335
mean_off_farming
mean_off_farming
0.2
0.8
0.7
0.01
1
NIL
HORIZONTAL

SLIDER
6
399
200
432
mean_age
mean_age
54
60
55.0
10
1
NIL
HORIZONTAL

SLIDER
5
215
197
248
threshold_in
threshold_in
0.5
0.8
0.6
0.1
1
NIL
HORIZONTAL

SLIDER
6
262
200
295
threshold_out
threshold_out
0.2
0.5
0.2
0.1
1
NIL
HORIZONTAL

SLIDER
7
451
202
484
average_degree
average_degree
5.68
14.37
8.62
0.01
1
NIL
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.1.1
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
